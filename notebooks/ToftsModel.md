Tofts Model
=================

The Tofts model describes has the following form:

$$
c(t) = K^{trans} \cdot c_a(t) \ast \exp(-k_{ep} \cdot t)
$$

where $c(t)$ is the tracer concentration in the tissue of interest, $c_a(t)$ is the arterial input function, $t$ is time, and $\ast$ denotes the convolution operator.
The fitting parameters are the transfer constant $K^{trans}$ and the rate constant $k_{ep}$ which is also equal to $K^{trans}/v_e}$ where $v_e$ is the extravascular extracellular space.
