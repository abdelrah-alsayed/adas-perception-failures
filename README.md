# adas-perception-failures

Explain why a pretrained ADAS perception model misses objects, using regional Shapley values (KernelSHAP).

Status: work in progress (setup phase).

## Reproduce

1. Clone the repository and go into the folder.
2. Create the environment:
   `conda env create -f environment.yml`
3. Activate it:
   `conda activate adas`
4. Run the tests:
   `python -m pytest`
   Expected output: `2 passed`