def gaussian_blurr_samples(samples, kernel, sigma):
    out = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(cv2.GaussianBlur(samples[i, j], (kernel, kernel), sigma))
        out.append(temp)

    return out
