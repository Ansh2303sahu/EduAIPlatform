# Threat Model (STRIDE-lite)

| Threat | Example | Mitigation |
|---|---|---|
| Spoofing | Fake identity | Auth + JWT |
| Tampering | Grade edits | RBAC + audit logs |
| Repudiation | Deny submission | immutable logs |
| Info Disclosure | View others | RLS + scoped APIs |
| DoS | upload spam | rate limiting |
| Privilege Escalation | student->admin | strict role checks |
