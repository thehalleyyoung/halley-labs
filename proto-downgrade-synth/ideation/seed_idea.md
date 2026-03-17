Community: area-088-security-privacy-and-cryptography

Idea: A symbolic-execution engine that extracts cipher-suite negotiation and handshake state machines from real TLS/SSH/QUIC library source code (C/Rust/Go), models Dolev-Yao network adversaries as SMT constraints, and automatically synthesizes concrete downgrade, version-rollback, and authentication-bypass attack traces—recovering known CVEs (FREAK, Logjam, POODLE) and discovering new negotiation flaws in current libraries.
