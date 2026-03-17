# Example: Backward Compatibility Verification

This example demonstrates verifying that backward-compatible protocol
negotiation does not introduce downgrade attack surfaces.

## Scenario

Many deployments must support legacy protocol versions alongside modern
ones. For example, an SSH server might support both the Terrapin-patched
"strict key exchange" extension and legacy key exchange for older clients.
NegSynth verifies that the coexistence of these codepaths does not allow
an attacker to force a downgrade.

## Running the Analysis

```bash
# Compile libssh2 with both legacy and modern paths enabled
CC=wllvm ./configure --enable-compat-mode
make
extract-bc src/.libs/libssh2.so -o libssh2.bc

# Run analysis targeting SSH extension negotiation
negsyn-cli analyze \
  --input libssh2.bc \
  --protocol ssh \
  --property extension-downgrade \
  --depth 25 \
  --budget 5 \
  --output results/ssh_compat.json

# Cross-library differential (if multiple SSH implementations available)
negsyn-cli diff \
  --libraries libssh2.bc,openssh.bc,dropbear.bc \
  --protocol ssh \
  --property negotiation-consistency \
  --output results/ssh_cross_library.json
```

## Expected Output for Terrapin (CVE-2023-48795)

When analyzing a pre-patch version of libssh2:
```
Attack Found: SSH extension negotiation manipulation
CVE: CVE-2023-48795 (Terrapin)
Severity: HIGH
Adversary budget used: 3 (sequence number manipulation)

Attack sequence:
  1. Intercept client SSH_MSG_KEXINIT
  2. Manipulate sequence numbers in Binary Packet Protocol
  3. Inject SSH_MSG_IGNORE to desynchronize sequence counters
  4. Force server to skip ext-info-s extension processing

Impact: Strict key exchange disabled, enabling prefix truncation attack
Trace: results/terrapin_attack.json
```

For a patched version:
```
Certificate: VALID
Property: No extension negotiation downgrade
Bounds: k=25, n=5
Coverage: 99.2% of reachable negotiation states
Strict-kex: Enforced when both endpoints support it
```

## Configuration

```toml
[analysis]
protocol = "ssh"
property = "extension-downgrade"

[bounds]
max_depth = 25
adversary_budget = 5

[ssh]
enable_strict_kex_check = true
track_sequence_numbers = true
```
