import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scipy.io as sio

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


def get_fixation_map(points, size=(768, 1024)):
    fixation_map = np.zeros((size[0] + 1, size[1] + 1), dtype=int)
    for point in points:
        x, y = point
        x_int = np.round(x).astype(int)
        y_int = np.round(y).astype(int)
        fixation_map[y_int, x_int] = 1
    return fixation_map


def get_CERF_data(path):
    IMG_DIR = path + "/faces/"
    fixationsMat = sio.loadmat(path + "/fixations.mat")
    imgListMat = sio.loadmat(path + "/imgList.mat")
    # load variables contained in mat files
    sbj = fixationsMat["sbj"]
    imgList = imgListMat["imgList"]
    numSub = 8
    numImg = len(imgList)
    stimuli = []
    to_exclude = []
    for i in range(numImg):
        try:
            img = plt.imread(IMG_DIR + imgList[i][0][0])
            stimuli.append(img)
        except:
            to_exclude.append(i)
            continue
    scans_per_img = {k: [] for k in range(numImg - len(to_exclude))}
    for s in range(numSub):
        k = 0
        for i in range(numImg):
            if i in to_exclude:
                continue
            x = sbj[0][s]["scan"][0][0][0][i]["fix_x"][0][0]
            y = sbj[0][s]["scan"][0][0][0][i]["fix_y"][0][0]
            dur = sbj[0][s]["scan"][0][0][0][i]["fix_duration"][0][0] / 1000.0
            curr_scan = np.hstack([x, y, dur])
            scans_per_img[k].append(curr_scan)
            k += 1
    # scans_per_img = {k: v for k, v in scans_per_img.items() if v}
    return stimuli, scans_per_img


def plot_scan_all_models(img_id, scan_idx, d, stimuli):
    from .vis import draw_scanpath

    # toplot = ["ittikoch", "geymol", "cle_dg", "ior_roi", "dg3", "tppgaze"]
    toplot = ["VQA", "ddm", "cle_dg", "ior_roi", "dg3", "tppgaze"]

    plt.figure(figsize=(15, 10))
    for i in range(0, 6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(stimuli)

        # import code; code.interact(local=locals())

        s = d[toplot[i]][img_id]
        if isinstance(d[toplot[i]][img_id], np.ndarray):
            s = [d[toplot[i]][img_id]]

        # import code; code.interact(local=locals())

        if len(s) > 1:
            scan = s[scan_idx]
        else:
            scan = s[0]
        if scan.shape[1] == 3:
            if toplot[i] == "tppgaze" or toplot[i] == "VQA" or toplot[i] == "ddm":
                draw_scanpath(scan[:, 0], scan[:, 1], scan[:, 2] * 1000)
            else:
                draw_scanpath(scan[:, 0], scan[:, 1], scan[:, 2])
        else:
            if "cle" in toplot[i]:
                scan = np.flip(scan, 1)
            draw_scanpath(scan[:, 0], scan[:, 1], np.ones(scan.shape[0]) * 100)
        plt.axis("off")
        plt.gca().set_title(toplot[i])
    # plt.show()
    # plt.savefig("/home/damelio/new_tpp_gaze/out.png")


def get_real_eye_data(fixations):
    omit_first = True  # omit first fixation
    if "pysaliency" in str(type(fixations)):
        imgIndexes = np.unique(fixations.n)
        scans_per_img = {k: [] for k in imgIndexes}
        for img_id in imgIndexes:
            f = fixations[fixations.n == img_id]
            subjects = np.unique(f.subjects)
            for s in subjects:
                curr_f = f[f.subjects == s]
                if curr_f.n.size < 4:
                    continue
                curr_scan = np.vstack([curr_f.x, curr_f.y, curr_f.duration]).T
                scans_per_img[img_id].append(curr_scan[omit_first:, :])
            # scans_per_img[img_id] = np.vstack([f.x, f.y, f.duration]).T
    else:
        scans_per_img = fixations
    # import code;code.interact(local=locals())
    return scans_per_img


def get_durs(d, to_seconds):
    durs = []
    for k, sbj in d.items():
        if type(sbj) != list:  # methods producing a single realisation
            sbj = [sbj]
        for scan in sbj:
            durs.append(scan[:, 2])
    if to_seconds:
        return np.concatenate(durs) / 1000.0
    return np.concatenate(durs)


def get_saccade_stats(d):
    import copy

    dirs = []
    amps = []
    n_imgs = len(d.keys())
    for ni in d.keys():
        if type(d[ni]) != list:
            d[ni] = [d[ni]]
        n_gen_sbj = len(d[ni])
        for k in range(n_gen_sbj):
            curr_scan = copy.deepcopy(d[ni][k])[:, 0:2]
            nfix = curr_scan.shape[0]
            for f in range(nfix - 1):
                curr_fix2 = curr_scan[f + 1, :]
                curr_fix1 = curr_scan[f, :]
                direction = (
                    np.arctan2(curr_fix2[1] - curr_fix1[1], curr_fix2[0] - curr_fix1[0])
                    + np.pi
                )
                amplitude = np.linalg.norm(curr_fix2 - curr_fix1)
                dirs.append(direction)
                amps.append(amplitude)
    return np.rad2deg(np.array(dirs)), np.array(amps)


def plot_scan_stats(d, m_to_compare):
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("talk")
    # noDurMethods = ['ittikoch','cle_dg', 'cle_cb', 'cle_itti', 'dg3']
    noDurMethods = ["ittikoch", "cle_dg", "cle_itti", "dg3"]
    durs = {k: [] for k in m_to_compare}
    dirs = {k: [] for k in m_to_compare}
    amps = {k: [] for k in m_to_compare}
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    c = {k: colors[i] for i, k in enumerate(d.keys())}
    names = {
        "Real": "Real",
        "tppgaze": "TPP-Gaze",
        "ior_roi": "IOR-ROI-LSTM",
        "geymol": "G-Eymol",
        "dg3": "DeepGazeIII",
        "ittikoch": "Itti&Koch",
        "cle_dg": "CLE (DG)",
        "VQA": "VQA",
        "ddm": "DDM",
    }
    plt.figure()
    for k in m_to_compare:
        ls = "-" if k == "Real" else "--"
        to_sec = k == "ior_roi" or k == "geymol"
        fill = False
        lw_real = 3.5
        lw_oth = 2
        if k not in noDurMethods:
            durs[k] = get_durs(d[k], to_seconds=to_sec)
            plt.subplot(1, 3, 1)
            if k == "Real":
                sns.kdeplot(
                    durs[k],
                    label=names[k],
                    color=c[k],
                    linestyle=ls,
                    fill=fill,
                    linewidth=lw_real,
                )
            else:
                sns.kdeplot(
                    durs[k],
                    label=names[k],
                    color=c[k],
                    linestyle=ls,
                    fill=fill,
                    linewidth=lw_oth,
                )
            plt.xlim(-0.2, 1.5)
            plt.gca().axes.yaxis.set_ticklabels([])
            # plt.xlabel('Fixations Duration [sec]')
            plt.gca().set_title("Fixations Duration [sec]")

        dirs[k], amps[k] = get_saccade_stats(d[k])
        plt.subplot(1, 3, 2)
        if k == "Real":
            sns.kdeplot(
                amps[k],
                label=names[k],
                color=c[k],
                linestyle=ls,
                fill=fill,
                linewidth=lw_real,
            )
        else:
            sns.kdeplot(
                amps[k],
                label=names[k],
                color=c[k],
                linestyle=ls,
                fill=fill,
                linewidth=lw_oth,
            )
        plt.xlim(-50.0, 1000.0)
        plt.gca().axes.yaxis.set_ticklabels([])
        # plt.xlabel('Saccades Amplitude [pix]')
        plt.gca().set_title("Saccades Amplitude [pix]")
        plt.legend()

        plt.subplot(1, 3, 3)
        if k == "Real":
            sns.kdeplot(
                dirs[k] - 180,
                label=names[k],
                color=c[k],
                linestyle=ls,
                fill=fill,
                linewidth=lw_real,
            )
        else:
            sns.kdeplot(
                dirs[k] - 180,
                label=names[k],
                color=c[k],
                linestyle=ls,
                fill=fill,
                linewidth=lw_oth,
            )
        plt.xticks(np.arange(-180, 181, 90))
        plt.xlim(-200.0, 200.0)
        plt.gca().axes.yaxis.set_ticklabels([])
        # plt.xlabel('Saccades Direction [deg]')
        plt.gca().set_title("Saccades Direction [deg]")
    # plt.show()

    # import code; code.interact(local=locals())


def plot_one_vs_real(img_id, d, stimuli, toplot="tppgaze"):
    import seaborn as sns

    # omit_first = True # omit first fixation
    sh = stimuli.shape
    # if "pysaliency" in str(type(fixations)):
    #     f = fixations[fixations.n == img_id]
    #     subjects = np.unique(f.subjects)
    #     scans = []
    #     for s in subjects:
    #         curr_f = f[f.subjects==s]
    #         if curr_f.n.size < 4:
    #             continue # skipping, not enough data
    #         curr_scan = np.vstack([curr_f.x, curr_f.y, curr_f.duration]).T
    #         scans.append(curr_scan[omit_first:,:])
    #     #scans = np.vstack([f.x, f.y, f.duration]).T
    # else:
    #     scans = fixations[img_id]

    # import code; code.interact(local=locals())
    realxy = np.vstack([s[:, :2] for s in d["Real"][img_id]])
    genxy = np.vstack([s[:, :2] for s in d[toplot][img_id]])
    real_heatmap = compute_density_image(realxy, size=(sh[0], sh[1]))
    gen_heatmap = compute_density_image(genxy, size=(sh[0], sh[1]))
    real_durs = np.concatenate([s[:, 2] for s in d["Real"][img_id]])
    gen_durs = np.concatenate([s[:, 2] for s in d["ddm"][img_id]])
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    ax = sns.kdeplot(np.squeeze(gen_durs), shade=True, color="b")
    ax = sns.kdeplot(np.squeeze(real_durs), shade=True, color="r")
    plt.legend(
        title="Duration Distributions", loc="upper right", labels=["Generated", "Real"]
    )
    plt.subplot(1, 3, 2)
    sns.set_style("white")
    plt.imshow(stimuli)
    plt.imshow(gen_heatmap, alpha=0.6)
    plt.axis("off")
    plt.title(toplot)
    plt.subplot(1, 3, 3)
    plt.imshow(stimuli)
    plt.imshow(real_heatmap, alpha=0.6)
    plt.title("Real")
    plt.axis("off")
    # plt.show()


def get_data(dataset="NUSEF"):
    import pysaliency

    data_location = "/home/damelio/tpp-gaze/pysaliency/data/"
    if dataset == "MIT":
        stimuli, fixations = (
            pysaliency.external_datasets.get_mit1003_with_initial_fixation(
                location=data_location
            )
        )
        # stimuli = stimuli.stimuli
    elif dataset == "NUSEF":
        stimuli, fixations = pysaliency.external_datasets.get_NUSEF_public(
            location=data_location
        )
        # stimuli = stimuli.stimuli
    elif dataset == "CERF":
        stimuli, fixations = get_CERF_data(
            "/home/damelio/tpp-gaze/data/CERF/CerfDataset"
        )
    elif dataset == "COCO_Search_18":
        stimuli, fixations = pysaliency.external_datasets.get_COCO_Search18(
            location=data_location
        )
    elif dataset == "COCO_Freeview":
        stimuli, fixations = pysaliency.external_datasets.get_COCO_Freeview(
            location=data_location
        )
    else:
        raise ValueError("Dataset not supported.")
    return stimuli, fixations


def image_process(image, img_resize=(32, 32), use_densenet=True):
    if not use_densenet:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    image = cv2.resize(
        image, (img_resize[1], img_resize[0]), interpolation=cv2.INTER_AREA
    )
    image = transform(image)
    return image


def get_heatmap(imgNum, dataset="mit", img_resize=16, fixations=None, stimuli=None):
    if dataset == "cerf":
        fixationsMat = sio.loadmat("../data/CERF/CerfDataset/fixations.mat")
        # load variables contained in mat files
        sbj = fixationsMat["sbj"]
        numSub = 8
        w = 1024.0
        h = 768.0
        data = []
        durs = []
        for ss in range(numSub):
            xx = sbj[0][ss]["scan"][0][0][0][imgNum]["fix_x"][0][0]
            yy = sbj[0][ss]["scan"][0][0][0][imgNum]["fix_y"][0][0]
            dur = sbj[0][ss]["scan"][0][0][0][imgNum]["fix_duration"][0][0] / 1000.0
            xx[xx > w] = w - 1
            yy[yy > h - 1] = h - 1
            xx[xx < 0] = 0
            yy[yy < 0] = 0
            xxyy = np.hstack([xx, yy])
            data.append(xxyy)
            durs.append(dur)
        datanp = np.vstack(data)
        heatmap = compute_density_image(datanp)
        if img_resize is not None:
            heatmap = cv2.resize(
                heatmap, (img_resize, img_resize), interpolation=cv2.INTER_AREA
            )
            heatmap = heatmap.reshape(1, -1)
    elif dataset == "mit":
        f = fixations[fixations.n == imgNum]
        datanp = np.vstack([f.x, f.y]).T
        sh = stimuli.stimuli[imgNum].shape
        heatmap = compute_density_image(datanp, size=(sh[0], sh[1]))
        durs = f.duration

        # import code; code.interact(local=locals())

    return heatmap, durs


def compute_density_image(points, size=(768, 1024)):
    # Saliency
    from scipy.ndimage import gaussian_filter

    points = np.flip(points, 1)
    sigma = 1 / 0.039
    H, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1], bins=(range(size[0] + 1), range(size[1] + 1))
    )
    Z = gaussian_filter(H, sigma=sigma)
    Z = Z / float(np.sum(Z))
    return Z


def clip_encode(img):
    import torch
    import open_clip
    from PIL import Image
    import torchvision.transforms as T

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    transform = T.ToPILImage()
    img = preprocess(transform(img)).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(img)

    return image_features


def get_img_scans_MIT(imgIdx):
    import pysaliency

    data_location = "/home/damelio/tpp-gaze/pysaliency/data/"
    mit_stimuli, mit_fixations = (
        pysaliency.external_datasets.get_mit1003_with_initial_fixation(
            location=data_location
        )
    )

    f = mit_fixations[mit_fixations.n == imgIdx]
    subjects = np.unique(f.subjects)

    scans = []
    for s in subjects:
        curr_f = f[f.subjects == s]
        curr_scan = np.vstack([curr_f.x, curr_f.y, curr_f.duration]).T
        scans.append(curr_scan)

    return scans


def plot_MIT_results(model, idx, d_test, n_simulations=100):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from dpp.data.batch import Batch
    import pysaliency

    a = []
    for dt in d_test:
        a.append(dt.img_id)
    available_idx = np.unique(a)

    assert idx >= 0 and idx < len(available_idx), (
        "idx exceeding the number of available test images ("
        + str(len(available_idx))
        + ")"
    )

    data_location = "/home/damelio/tpp-gaze/pysaliency/data/"

    mit_stimuli, mit_fixations = (
        pysaliency.external_datasets.get_mit1003_with_initial_fixation(
            location=data_location
        )
    )

    imgIdx = available_idx[idx]
    img = mit_stimuli.stimuli[imgIdx]
    sh = mit_stimuli.stimuli[imgIdx].shape
    test_img = image_process(img, img_resize=(512, 512))
    bs = n_simulations
    start_it = torch.zeros([bs, 1]) * 1e-10
    start_mark = torch.randn(bs, 1, 2)
    start_mask = torch.ones([bs, 1])
    start_stimuli = test_img[None, :, :, :].repeat(bs, 1, 1, 1)  # RAW
    start_batch = Batch(
        inter_times=start_it, mask=start_mask, marks=start_mark, stimuli=start_stimuli
    )
    model.eval()
    sampled_batch = model.sample(
        t_end=3.0, batch_size=bs, img=test_img, start_batch=start_batch
    )
    hm, durs = get_heatmap(
        imgIdx, stimuli=mit_stimuli, fixations=mit_fixations, img_resize=None
    )
    sns.set_style("whitegrid")
    plt.figure(figsize=(12.8, 9.6))
    plt.subplot(1, 3, 1)
    sd = sampled_batch.inter_times[sampled_batch.mask.bool()].reshape(-1).cpu().numpy()
    # rd = np.concatenate(durs).squeeze()
    rd = durs
    sns.kdeplot(sd, label="Sampled", alpha=0.5, color="r")
    sns.kdeplot(rd, label="Real", alpha=0.5, color="g")
    plt.legend()
    msk = np.stack(
        [sampled_batch.mask.cpu().numpy(), sampled_batch.mask.cpu().numpy()], axis=2
    )
    dim = 16
    w = sh[1]
    h = sh[0]
    mrks = sampled_batch.marks[sampled_batch.mask.bool()].cpu().numpy().astype(float)
    xy_np = mrks
    xy_np[:, 0] = ((xy_np[:, 0] + 1) / 2) * w
    xy_np[:, 1] = ((xy_np[:, 1] + 1) / 2) * h
    sns.set_style("white")
    z = compute_density_image(xy_np, size=(h, w))
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(z, alpha=0.6)
    plt.title("Sampled")
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(hm, alpha=0.6)
    plt.title("Real")

    plt.savefig(
        "/home/damelio/new_tpp_gaze/data/figures/stimId_" + str(imgIdx) + ".png"
    )

    return sampled_batch


def plot_results(
    idx, model, d_train, d_test, n_simulations=100, train_or_test="test", img_size=224
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from dpp.data.batch import Batch

    IMG_DIR = "../data/CERF/CerfDataset/faces/"
    imgListMat = sio.loadmat("../data/CERF/CerfDataset/imgList.mat")
    imgList = imgListMat["imgList"]
    if train_or_test == "train":
        imgIdx = d_train[idx].img_id
    else:
        imgIdx = d_test[idx].img_id
    img = plt.imread(IMG_DIR + imgList[imgIdx][0][0])
    test_img = image_process(img, img_resize=(img_size, img_size))
    bs = n_simulations
    start_it = torch.zeros([bs, 1]) * 1e-10
    start_mark = torch.randn(bs, 1, 2)
    start_mask = torch.ones([bs, 1])
    start_stimuli = test_img[None, :, :, :].repeat(bs, 1, 1, 1)  # RAW
    start_batch = Batch(
        inter_times=start_it, mask=start_mask, marks=start_mark, stimuli=start_stimuli
    )
    model.eval()
    sampled_batch = model.sample(
        t_end=2.0, batch_size=bs, img=test_img, start_batch=start_batch
    )
    hm, durs = get_heatmap(imgIdx, img_resize=None)
    sns.set_style("whitegrid")
    plt.figure()
    plt.subplot(1, 3, 1)
    sd = sampled_batch.inter_times[sampled_batch.mask.bool()].reshape(-1).cpu().numpy()
    rd = np.concatenate(durs).squeeze()
    sns.kdeplot(sd, label="Sampled", alpha=0.5, color="r")
    sns.kdeplot(rd, label="Real", alpha=0.5, color="g")
    plt.legend()
    msk = np.stack(
        [sampled_batch.mask.cpu().numpy(), sampled_batch.mask.cpu().numpy()], axis=2
    )
    dim = 16
    w = 1024.0
    h = 768.0
    mrks = sampled_batch.marks[sampled_batch.mask.bool()].cpu().numpy().astype(float)
    xy_np = mrks
    xy_np[:, 0] = ((xy_np[:, 0] + 1) / 2) * w
    xy_np[:, 1] = ((xy_np[:, 1] + 1) / 2) * h
    sns.set_style("white")
    z = compute_density_image(xy_np)
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(z, alpha=0.6)
    plt.title("Sampled")
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(hm, alpha=0.6)
    plt.title("Real")
    return sampled_batch
