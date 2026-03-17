# Example: TLS Version Migration Analysis

This example demonstrates using NegSynth to verify that a TLS library correctly
handles version migration from TLS 1.2 to TLS 1.3 without introducing downgrade
attack vectors.

## Scenario

When a library adds TLS 1.3 support while maintaining TLS 1.2 backward compatibility,
the version negotiation logic must ensure that an active attacker cannot force
endpoints that both support TLS 1.3 to negotiate TLS 1.2 instead.

## Running the Analysis

```bash
# Step 1: Compile the target library to LLVM bitcode
export LLVM_COMPILER=clang
CC=wllvm ./config enable-tls1_3
make -j$(nproc)
extract-bc libssl.so -o libssl.bc

# Step 2: Run NegSynth analysis focused on version negotiation
negsyn-cli analyze \
  --input libssl.bc \
  --protocol tls \
  --property version-downgrade \
  --min-version tls12 \
  --max-version tls13 \
  --depth 20 \
  --budget 5 \
  --output results/tls13_migration.json

# Step 3: Inspect results
negsyn-cli inspect --input results/tls13_migration.json
```

## Expected Results

For a correctly implemented library (e.g., OpenSSL 3.2):
```
Certificate: VALID
Property: No TLS 1.3 → TLS 1.2 version downgrade
Bounds: k=20, n=5
Coverage: 99.7% of reachable version-negotiation states
Sentinel check: TLS 1.3 anti-downgrade sentinel verified
```

For a vulnerable library version:
```
Attack Found: Version downgrade TLS 1.3 → TLS 1.2
Adversary steps: 3
Attack vector: ServerHello version field manipulation
Severity: HIGH
Trace: results/version_downgrade_attack.json
```

## Key Configuration

```toml
[analysis]
protocol = "tls"
property = "version-downgrade"

[bounds]
max_depth = 20
adversary_budget = 5

[merge]
strategy = "algebraic"
enable_version_lattice = true

[encoding]
theory = "BV+Arrays+UF+LIA"
adversary_model = "dolev-yao"
```
