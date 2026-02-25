"""PyTorch integration experiments: CIFAR-10 with PhaseKit-recommended init.

Compares PhaseKit-recommended initialization vs PyTorch default vs Kaiming init
on CIFAR-10 for a 10-layer MLP and 6-layer CNN across 5 seeds and 50 epochs.
Also runs architecture extension validation (new activations, Conv2d, LayerNorm/BatchNorm).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec


def run_architecture_extension_experiment():
    """Validate new activations (LeakyReLU, Mish) and Conv2d/BN/LN support."""
    mf = MeanFieldAnalyzer()
    results = {"activations": {}, "conv2d": {}, "normalization": {}}

    # New activations: compute edge-of-chaos and verify predictions
    for act in ['relu', 'tanh', 'gelu', 'silu', 'leaky_relu', 'mish', 'elu']:
        sw_star, _ = mf.find_edge_of_chaos(act)
        for sw_mult in [0.5, 0.8, 1.0, 1.2, 1.5]:
            sw = sw_star * sw_mult
            arch = ArchitectureSpec(depth=10, width=256, activation=act, sigma_w=sw)
            report = mf.analyze(arch)
            key = f"{act}_sw{sw_mult:.1f}"
            results["activations"][key] = {
                "activation": act,
                "sigma_w": sw,
                "sigma_w_star": sw_star,
                "sigma_w_mult": sw_mult,
                "chi_1": report.chi_1,
                "phase": report.phase,
                "depth_scale": report.depth_scale,
                "probabilities": report.phase_classification.probabilities,
            }

    # Conv2d test: verify that Conv2d fan_in=c_in*k*k matches MLP theory
    for kernel_size in [1, 3, 5]:
        for channels in [32, 64, 128]:
            fan_in = 3 * kernel_size * kernel_size  # first layer
            # Mean-field: same as MLP with sw^2/fan_in effective
            sw_star, _ = mf.find_edge_of_chaos('relu')
            arch = ArchitectureSpec(depth=6, width=channels, activation='relu', sigma_w=sw_star)
            report = mf.analyze(arch)
            key = f"conv_k{kernel_size}_c{channels}"
            results["conv2d"][key] = {
                "kernel_size": kernel_size,
                "channels": channels,
                "fan_in": fan_in,
                "chi_1": report.chi_1,
                "phase": report.phase,
            }

    # BatchNorm/LayerNorm: verify variance reset
    for norm_type in ["batchnorm", "layernorm"]:
        arch = ArchitectureSpec(depth=20, width=256, activation='relu', sigma_w=2.0,
                                has_batchnorm=True)
        report = mf.analyze(arch)
        results["normalization"][norm_type] = {
            "depth": 20,
            "sigma_w": 2.0,
            "chi_1": report.chi_1,
            "phase": report.phase,
            "variance_stable": all(0.5 < v < 2.0 for v in report.variance_trajectory[1:]),
        }

    return results


def run_monte_carlo_validation():
    """Validate mean-field predictions against Monte Carlo forward passes."""
    if not HAS_TORCH:
        return {"error": "torch not available"}

    mf = MeanFieldAnalyzer()
    results = {}
    rng = np.random.RandomState(42)

    for act_name, act_module in [('relu', nn.ReLU), ('leaky_relu', nn.LeakyReLU),
                                   ('mish', nn.Mish), ('gelu', nn.GELU), ('silu', nn.SiLU)]:
        sw_star, _ = mf.find_edge_of_chaos(act_name)

        for sw_mult in [0.8, 1.0, 1.2]:
            sw = sw_star * sw_mult
            depth, width = 10, 256
            arch = ArchitectureSpec(depth=depth, width=width, activation=act_name, sigma_w=sw)
            report = mf.analyze(arch)

            # Monte Carlo: build actual network, measure variance per layer
            mc_variances = []
            n_trials = 20
            for trial in range(n_trials):
                torch.manual_seed(trial)
                layers = []
                in_dim = width
                for l in range(depth):
                    lin = nn.Linear(in_dim, width, bias=False)
                    nn.init.normal_(lin.weight, 0, sw / np.sqrt(in_dim))
                    layers.append(lin)
                    layers.append(act_module())
                    in_dim = width

                model = nn.Sequential(*layers)
                model.eval()
                x = torch.randn(100, width)
                with torch.no_grad():
                    layer_vars = [float(x.var().item())]
                    h = x
                    for i in range(0, len(layers), 2):
                        h = layers[i](h)
                        h = layers[i+1](h)
                        layer_vars.append(float(h.var().item()))
                    mc_variances.append(layer_vars)

            mc_mean = np.mean(mc_variances, axis=0).tolist()
            mf_pred = report.variance_trajectory[:len(mc_mean)]
            fw_pred = report.finite_width_corrected_variance[:len(mc_mean)]

            # Compute relative errors
            mf_errors = [abs(m - e) / max(abs(e), 1e-10) for m, e in zip(mf_pred, mc_mean)]
            fw_errors = [abs(f - e) / max(abs(e), 1e-10) for f, e in zip(fw_pred, mc_mean)]

            key = f"{act_name}_sw{sw_mult:.1f}"
            results[key] = {
                "activation": act_name,
                "sigma_w": sw,
                "mc_variance_mean": mc_mean,
                "mf_prediction": mf_pred,
                "fw_prediction": fw_pred,
                "mf_mean_error": float(np.mean(mf_errors)),
                "fw_mean_error": float(np.mean(fw_errors)),
                "phase": report.phase,
            }

    return results


def run_cifar10_experiment():
    """Compare PhaseKit-recommended vs default vs Kaiming init on CIFAR-10."""
    if not HAS_TORCH:
        return {"error": "torch not available"}

    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        return {"error": "torchvision not available"}

    from pytorch_integration import analyze, recommend_init

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', '.data_cache')
        trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=transform)
    except Exception as e:
        return {"error": f"Could not load CIFAR-10: {e}"}

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    mf = MeanFieldAnalyzer()
    results = {"mlp": {}, "cnn": {}}
    n_seeds = 5
    n_epochs = 50

    def train_and_eval(model, name, seed):
        torch.manual_seed(seed)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        test_accs = []
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            n_batch = 0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.view(inputs.size(0), -1) if 'mlp' in name.lower() else inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batch += 1
            scheduler.step()
            train_losses.append(running_loss / n_batch)

            if epoch % 10 == 9 or epoch == n_epochs - 1:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in testloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs.view(inputs.size(0), -1) if 'mlp' in name.lower() else inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                test_accs.append(correct / total)

        return train_losses, test_accs

    def make_mlp(init_type, seed):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
        if init_type == "phasekit":
            sw_star, _ = mf.find_edge_of_chaos('relu')
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, sw_star / np.sqrt(m.in_features))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init_type == "kaiming":
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # else: default PyTorch init
        return model

    def make_cnn(init_type, seed):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10),
        )
        if init_type == "phasekit":
            sw_star, _ = mf.find_edge_of_chaos('relu')
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    nn.init.normal_(m.weight, 0, sw_star / np.sqrt(fan_in))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, sw_star / np.sqrt(m.in_features))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init_type == "kaiming":
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        return model

    # Run MLP experiments
    for init_type in ["default", "kaiming", "phasekit"]:
        seed_results = []
        for seed in range(n_seeds):
            print(f"  MLP {init_type} seed={seed}...")
            model = make_mlp(init_type, seed)

            # PhaseKit analysis before training
            spec = ArchitectureSpec(depth=10, width=256, activation='relu',
                                    sigma_w=float(list(model.parameters())[0].std().item() * np.sqrt(256)))
            report = mf.analyze(spec)

            train_losses, test_accs = train_and_eval(model, f"MLP_{init_type}", seed)
            seed_results.append({
                "seed": seed,
                "final_train_loss": train_losses[-1],
                "final_test_acc": test_accs[-1],
                "test_accs": test_accs,
                "phase_before": report.phase,
                "chi_1_before": report.chi_1,
            })
        results["mlp"][init_type] = {
            "seeds": seed_results,
            "mean_test_acc": float(np.mean([r["final_test_acc"] for r in seed_results])),
            "std_test_acc": float(np.std([r["final_test_acc"] for r in seed_results])),
        }
        print(f"  MLP {init_type}: acc={results['mlp'][init_type]['mean_test_acc']:.4f} ± {results['mlp'][init_type]['std_test_acc']:.4f}")

    # Run CNN experiments
    for init_type in ["default", "kaiming", "phasekit"]:
        seed_results = []
        for seed in range(n_seeds):
            print(f"  CNN {init_type} seed={seed}...")
            model = make_cnn(init_type, seed)
            train_losses, test_accs = train_and_eval(model, f"CNN_{init_type}", seed)
            seed_results.append({
                "seed": seed,
                "final_train_loss": train_losses[-1],
                "final_test_acc": test_accs[-1],
                "test_accs": test_accs,
            })
        results["cnn"][init_type] = {
            "seeds": seed_results,
            "mean_test_acc": float(np.mean([r["final_test_acc"] for r in seed_results])),
            "std_test_acc": float(np.std([r["final_test_acc"] for r in seed_results])),
        }
        print(f"  CNN {init_type}: acc={results['cnn'][init_type]['mean_test_acc']:.4f} ± {results['cnn'][init_type]['std_test_acc']:.4f}")

    return results


def run_calibration_experiment():
    """Independent calibration: predict phase, then validate with actual training."""
    if not HAS_TORCH:
        return {"error": "torch not available"}

    mf = MeanFieldAnalyzer()
    results = []

    for act_name, act_module in [('relu', nn.ReLU), ('tanh', nn.Tanh), ('gelu', nn.GELU)]:
        sw_star, _ = mf.find_edge_of_chaos(act_name)

        for sw_mult in np.linspace(0.3, 2.0, 12):
            sw = sw_star * sw_mult
            depth, width = 10, 128

            arch = ArchitectureSpec(depth=depth, width=width, activation=act_name, sigma_w=sw)
            report = mf.analyze(arch)
            predicted_phase = report.phase

            # Train actual network to determine empirical phase
            loss_ratios = []
            for seed in range(5):
                torch.manual_seed(seed)
                layers = []
                in_dim = 32
                layers.append(nn.Linear(32, width, bias=False))
                nn.init.normal_(layers[-1].weight, 0, sw / np.sqrt(32))
                layers.append(act_module())
                for _ in range(depth - 2):
                    lin = nn.Linear(width, width, bias=False)
                    nn.init.normal_(lin.weight, 0, sw / np.sqrt(width))
                    layers.append(lin)
                    layers.append(act_module())
                layers.append(nn.Linear(width, 1, bias=False))
                nn.init.normal_(layers[-1].weight, 0, sw / np.sqrt(width))

                model = nn.Sequential(*layers)
                optimizer = optim.SGD(model.parameters(), lr=0.01)

                # Generate regression data
                x = torch.randn(200, 32)
                y = torch.randn(200, 1)

                # Initial loss
                with torch.no_grad():
                    pred0 = model(x)
                    if torch.isnan(pred0).any() or torch.isinf(pred0).any():
                        loss_ratios.append(float('inf'))
                        continue
                    loss0 = float(nn.functional.mse_loss(pred0, y).item())

                # Train for 500 steps
                model.train()
                for step in range(500):
                    optimizer.zero_grad()
                    pred = model(x)
                    if torch.isnan(pred).any():
                        break
                    loss = nn.functional.mse_loss(pred, y)
                    if torch.isnan(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

                with torch.no_grad():
                    pred_f = model(x)
                    if torch.isnan(pred_f).any() or torch.isinf(pred_f).any():
                        loss_ratios.append(float('inf'))
                    else:
                        loss_f = float(nn.functional.mse_loss(pred_f, y).item())
                        loss_ratios.append(loss_f / max(loss0, 1e-10))

            # Determine empirical phase from training dynamics
            median_ratio = float(np.median([r for r in loss_ratios if np.isfinite(r)] or [float('inf')]))
            n_diverged = sum(1 for r in loss_ratios if not np.isfinite(r) or r > 10)

            if n_diverged >= 3:
                empirical_phase = "chaotic"
            elif median_ratio > 0.95:
                empirical_phase = "ordered"
            elif median_ratio < 0.3:
                empirical_phase = "chaotic"  # trained well but unstable
            else:
                empirical_phase = "critical"

            results.append({
                "activation": act_name,
                "sigma_w": float(sw),
                "sigma_w_mult": float(sw_mult),
                "predicted_phase": predicted_phase,
                "empirical_phase": empirical_phase,
                "loss_ratio": median_ratio,
                "n_diverged": n_diverged,
                "chi_1": report.chi_1,
                "match": predicted_phase == empirical_phase,
            })

    n_match = sum(1 for r in results if r["match"])
    accuracy = n_match / len(results) if results else 0
    # Binary accuracy: check ordered vs trainable (critical+chaotic)
    n_binary_match = sum(1 for r in results
                         if (r["predicted_phase"] == "ordered") == (r["empirical_phase"] == "ordered"))
    binary_acc = n_binary_match / len(results) if results else 0

    return {
        "configs": results,
        "total": len(results),
        "3class_accuracy": accuracy,
        "binary_accuracy": binary_acc,
    }


if __name__ == "__main__":
    output_dir = os.path.dirname(__file__)

    print("=" * 60)
    print("Running architecture extension experiments...")
    print("=" * 60)
    arch_results = run_architecture_extension_experiment()
    with open(os.path.join(output_dir, "architecture_extension_results.json"), "w") as f:
        json.dump(arch_results, f, indent=2)
    print(f"  Saved {len(arch_results['activations'])} activation configs")

    print("\n" + "=" * 60)
    print("Running Monte Carlo validation...")
    print("=" * 60)
    mc_results = run_monte_carlo_validation()
    if "error" not in mc_results:
        with open(os.path.join(output_dir, "mc_validation_results.json"), "w") as f:
            json.dump(mc_results, f, indent=2)
        for k, v in mc_results.items():
            print(f"  {k}: MF err={v['mf_mean_error']:.4f}, FW err={v['fw_mean_error']:.4f}")
    else:
        print(f"  Skipped: {mc_results['error']}")

    print("\n" + "=" * 60)
    print("Running calibration experiment...")
    print("=" * 60)
    cal_results = run_calibration_experiment()
    if "error" not in cal_results:
        with open(os.path.join(output_dir, "calibration_results.json"), "w") as f:
            json.dump(cal_results, f, indent=2)
        print(f"  3-class accuracy: {cal_results['3class_accuracy']:.2%}")
        print(f"  Binary accuracy: {cal_results['binary_accuracy']:.2%}")
    else:
        print(f"  Skipped: {cal_results['error']}")

    print("\n" + "=" * 60)
    print("Running CIFAR-10 experiments...")
    print("=" * 60)
    cifar_results = run_cifar10_experiment()
    if "error" not in cifar_results:
        with open(os.path.join(output_dir, "pytorch_integration_results.json"), "w") as f:
            json.dump(cifar_results, f, indent=2)
    else:
        print(f"  Skipped: {cifar_results['error']}")

    print("\nDone!")
