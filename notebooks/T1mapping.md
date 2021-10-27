T1 mapping
=================

The MRI signal in a spoiled gradient recalled echo is described by:

$$
S = M_0 \sin(\alpha) \frac{1 - \exp(-TR/T_1)}{1 - \cos(\alpha) \exp(-TR/T_1)} \exp(-TE/T_2)
$$

The acquisition parameters are the flip angle $\alpha$, repetition time $TR$ and echo time $TE$---these are controlled by the scanner and are known[^1].
The goal of T1 mapping is to estimate the tissue-specific parameters---the magnetization $M_0$ and spin-lattice relaxation time $T_1$.

[^1]: Let's ignore B1 effects for now.
