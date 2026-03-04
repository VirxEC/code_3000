# Security Overview

## Intended Users
This repository is intended for students, instructors, and reviewers involved in CSE 3000: Contemporary Issues in Computer Science and Engineering. It contains assignment code, notebooks, and related datasets used for coursework and evaluation. All datasets in this repository are synthetic/fake.

## Risk Assessment
The primary risks if the code or data fell into the wrong hands include:
- **Academic integrity concerns**: Assignment solutions could be misused for plagiarism or unauthorized distribution.
- **Model misuse**: Scripts or notebooks could be adapted to perform deanonymization or profiling outside of approved contexts.

Because the datasets are synthetic/fake, privacy or sensitive data exposure is not a concern. Overall risk is **low**: while no production systems are present, the educational materials could still be misused if redistributed without safeguards.

## Security Measures
The following steps are in place:
- **`CODEOWNERS`** is present to clarify review ownership and accountability.
- The repository is intended for internal educational use; no deployment credentials or secrets are stored here.
- Sensitive data (if any) should remain in restricted directories and excluded from public sharing.

If additional controls are required (e.g., protected branches or PR rulesets), they can be configured in the hosting platform to enforce reviews and limit write access.