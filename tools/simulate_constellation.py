import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_constellation(
    modulation: str = "16qam",
    snr_db: float = 15.0,
    n_symbols: int = 2000,
    out_dir: str = "outputs"
):
    """
    Generates constellation scatter plot for a given modulation across AWGN.

    Returns:
      {
        "plots": [<png path>],
        "kpis": {"snr_db": snr_db, "modulation": modulation}
      }
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        import tensorflow as tf
        from sionna.mapping import Mapper, Constellation
        from sionna.channel import AWGN
    except Exception as e:
        return {
            "plots": [],
            "kpis": {},
            "error": f"Sionna/TensorFlow not available: {e}"
        }

    mod = modulation.lower()
    # Supported: qpsk, 16qam, 64qam, 256qam
    if mod == "qpsk":
        constellation = Constellation("qam", 4)
    elif mod.endswith("qam"):
        m = int(mod.replace("qam", ""))
        constellation = Constellation("qam", m)
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation=constellation)
    awgn = AWGN()

    # Random bits -> symbols
    n_bits_per_sym = int(np.log2(constellation.num_points))
    bits = tf.random.uniform([n_symbols, n_bits_per_sym], minval=0, maxval=2, dtype=tf.int32)
    x = mapper(bits)

    # AWGN
    snr_lin = 10 ** (snr_db / 10)
    noise_var = 1 / snr_lin
    y = awgn([x, tf.constant(noise_var, tf.float32)])

    y_np = y.numpy().reshape(-1)

    # Plot
    fig = plt.figure()
    plt.scatter(np.real(y_np), np.imag(y_np), s=6, alpha=0.5)
    plt.title(f"{modulation.upper()} Constellation @ SNR={snr_db} dB")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    plot_path = os.path.join(out_dir, f"constellation_{mod}_{snr_db}db.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {"snr_db": snr_db, "modulation": modulation, "n_symbols": n_symbols}
    }
