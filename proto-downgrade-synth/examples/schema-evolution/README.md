# Example: Cipher Suite Schema Evolution

This example shows how NegSynth handles evolution of cipher suite schemas
across library versions, detecting when added or removed cipher suites
introduce downgrade paths.

## Scenario

A library removes support for weak cipher suites (e.g., RC4, 3DES) in a
new version. The negotiation fallback logic must not create a path where
an attacker can force selection of a remaining weak suite when strong
alternatives are available.

## Running the Analysis

```bash
# Analyze the old version
negsyn-cli analyze \
  --input libssl_old.bc \
  --protocol tls \
  --property cipher-downgrade \
  --output results/old_version.json

# Analyze the new version
negsyn-cli analyze \
  --input libssl_new.bc \
  --protocol tls \
  --property cipher-downgrade \
  --output results/new_version.json

# Differential analysis between versions
negsyn-cli diff \
  --old results/old_version.json \
  --new results/new_version.json \
  --output results/evolution_diff.json
```

## Expected Output

```
Differential Analysis: libssl v1.1.1 → v3.2.0

Cipher suites removed: 23 (12 weak, 8 medium, 3 strong)
Cipher suites added: 5 (all strong)
Negotiation paths changed: 47

New downgrade paths: 0
Eliminated downgrade paths: 8
Remaining downgrade paths: 2 (pre-existing, LOW severity)

Verdict: Schema evolution is SAFE
  - No new downgrade vectors introduced
  - 8 previously known vectors eliminated by suite removal
  - 2 remaining vectors involve TLS_RSA_WITH_AES_128_CBC_SHA256
    (classified LOW: requires RSA key compromise)
```

## Key Points

1. **Differential mode** compares state machines extracted from two versions
2. **Cipher suite tracking** monitors which suites are offered and selected
3. **Impact scoring** classifies severity based on cryptographic strength delta
4. **Regression detection** catches new downgrade paths introduced by changes
