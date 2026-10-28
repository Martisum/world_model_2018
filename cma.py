import numpy as np


def cma_es(f, x0, sigma0, maxiter=200, seed=0, verbose=True):
    """标准 CMA-ES（无边界处理）。f 越大越好。"""
    n = len(x0)
    lam = 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    rng = np.random.default_rng(seed)
    m = np.array(x0, float)
    sigma = float(sigma0)
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)

    for g in range(maxiter):
        # ① 采样
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        Z = rng.standard_normal((lam, n))
        Y = (Z * D) @ B.T              # 每行 y_i ~ N(0, C)
        X = m + sigma * Y
        F = np.array([f(x) for x in X])

        # ② 排序
        idx = np.argsort(-F)
        Xs, Ys = X[idx], Y[idx]

        # ③ 重组均值
        m_old = m.copy()
        m = w @ Xs[:mu]
        y_w = (m - m_old) / sigma

        # ④ 协方差
        Cinv_sqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ y_w)
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1)))
                < (1.4 + 2 / (n + 1)) * chi)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (Ys[:mu].T * w) @ Ys[:mu])
        C = (C + C.T) / 2

        # ⑤ 步长
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))

        if verbose:
            if n <= 10:
                print(f"g={g:3d} m={np.round(m, 4)} sigma={sigma:.4f} "
                      f"f(m)={f(m):.3e}")
            else:
                # 高维控制器（例如 867 维）打印整个向量没有意义
                print(f"g={g:3d} sigma={sigma:.4f} f(m)={f(m):.3e}")

    return m, sigma, C


# ---------- 奶茶游戏 ----------
def milk_tea(x):
    return -((x[0] + x[1] - 4) ** 2 + 0.02 * (x[0] - x[1]) ** 2)


if __name__ == "__main__":
    m, s, C = cma_es(milk_tea, [0.0, 0.0], 1.5, maxiter=60, seed=0)
    print("m =", np.round(m, 4))
    print("sigma =", round(s, 4))
    print("f(m) =", milk_tea(m))
    print("C =\n", np.round(C, 4))