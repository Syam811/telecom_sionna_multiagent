import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_ber(
    modulation: str = "qpsk",
    channel: str = "awgn",         # "awgn" or "rayleigh"
    snr_db_list = None,           # e.g., [-5,0,5,10,15]
    n_bits: int = 200000,
    batch_size: int = 2000,
    out_dir: str = "outputs"
):
    """
    BER vs SNR for a given modulation and channel.

    Returns:
      {
        "plots": [<png path>],
        "kpis": {"snr_db": [...], "ber": [...], "modulation": ..., "channel": ...}
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    if snr_db_list is None:
        snr_db_list = [-5, 0, 5, 10, 15]

    try:
        import tensorflow as tf
        from sionna.mapping import Mapper, Demapper, Constellation
        from sionna.channel import AWGN, FlatFadingChannel
        from sionna.utils import ebnodb2no
    except Exception as e:
        return {"plots": [], "kpis": {}, "error": f"Sionna/TensorFlow not available: {e}"}

    mod = modulation.lower()
    if mod == "qpsk":
        constellation = Constellation("qam", 4)
    elif mod.endswith("qam"):
        m = int(mod.replace("qam", ""))
        constellation = Constellation("qam", m)
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation)

    chan = channel.lower()
    if chan == "awgn":
        channel_layer = AWGN()
        fading = False
    elif chan == "rayleigh":
        channel_layer = FlatFadingChannel(num_tx_ant=1, num_rx_ant=1, add_awgn=True)
        fading = True
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown channel: {channel}"}

    n_bits_per_sym = int(np.log2(constellation.num_points))
    bers = []

    for snr_db in snr_db_list:
        no = ebnodb2no(snr_db, n_bits_per_sym, coderate=1.0)
        n_errors = 0
        n_total = 0

        while n_total < n_bits:
            b = tf.random.uniform([batch_size, n_bits_per_sym], 0, 2, dtype=tf.int32)
            x = mapper(b)

            if fading:
                # FlatFadingChannel expects [x, no]
                y, h = channel_layer([x, no])
                llr = demapper([y, h, no])
            else:
                y = channel_layer([x, no])
                llr = demapper([y, no])

            b_hat = tf.cast(llr > 0, tf.int32)
            n_errors += tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat), tf.int32)).numpy()
            n_total += batch_size * n_bits_per_sym

        ber = n_errors / n_total
        bers.append(ber)

    # Plot
    fig = plt.figure()
    plt.semilogy(snr_db_list, bers, marker="o")
    plt.title(f"BER vs SNR ({modulation.upper()} - {channel.upper()})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")

    plot_path = os.path.join(out_dir, f"ber_{mod}_{chan}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {"snr_db": snr_db_list, "ber": bers, "modulation": modulation, "channel": channel}
    }
