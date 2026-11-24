# results/

Generated, not committed.

`kernelforge tune` and `kernelforge benchmark` write `kernelforge.db` here, and
`kernelforge report` reads it to produce `reports/`. Both are gitignored.

Measurements belong to the machine that produced them: an RTX 4090 number and
an A100 number in the same table are not a comparison. Committing a database
would mix one card's results into another's and silently invalidate every
aggregate computed from it. The database records the GPU, driver, CUDA,
PyTorch and Triton versions for each run, so if you do share one, that context
travels with it.

A database still accumulates runs across a driver or PyTorch upgrade, so that
context is also what `kernelforge report` and `ResultsDB.best_per_label` group
by: each distinct environment is numbered, gets its own row in every table, and
is never ranked against or divided into another. Mixing is reported rather
than resolved by taking whichever number happened to be lower.

To produce results:

```bash
kernelforge tune matmul --m 2048 --n 4096 --k 4096 --dtype fp16
kernelforge benchmark rmsnorm --suite sweep --dtype fp16
kernelforge report
```
