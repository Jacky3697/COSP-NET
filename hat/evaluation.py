import sys
sys.path.append('../common')

from common import utils, metrics
from torch.distributions import Categorical
import torch, json
from tqdm import tqdm
import numpy as np
from os.path import join
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import gaussian_filter
try:
    from multimatch import multimatch as mmg
    HAS_MULTIMATCH = True
except ImportError:
    HAS_MULTIMATCH = False
    print("[WARN] multimatch not found, skip MultiMatch")

class NumpyEncoder(json.JSONEncoder):
    """自定义编码器，处理 NumPy 数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def get_IOR_mask(norm_x, norm_y, h, w, r):
    bs = len(norm_x)
    x, y = norm_x * w, norm_y * h
    Y, X = np.ogrid[:h, :w]
    X = X.reshape(1, 1, w)
    Y = Y.reshape(1, h, 1)
    x = x.reshape(bs, 1, 1)
    y = y.reshape(bs, 1, 1)
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return torch.from_numpy(mask.reshape(bs, -1))


def scanpath_decode(model, img, task_ids, pa, sample_action=False, center_initial=True):

    bs = img.size(0)
    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(img)
    if center_initial:
        normalized_fixs = torch.zeros(bs, 1, 2).fill_(0.5)
        action_mask = get_IOR_mask(np.ones(bs) * 0.5,
                                   np.ones(bs) * 0.5,
                                   pa.im_h, 
                                   pa.im_w, 
                                   pa.IOR_radius)
    else:
        normalized_fixs = torch.zeros(bs, 0, 2)
        action_mask = torch.zeros(bs, pa.im_h * pa.im_w)
        
    stop_flags = []
    for i in range(pa.max_traj_length):
        with torch.no_grad():
            if i == 0 and not center_initial:
                ys = ys_high = torch.zeros(bs, 1).to(torch.long)
                padding = torch.ones(bs, 1).bool().to(img.device)
            else:
                ys, ys_high = utils.transform_fixations(
                    normalized_fixs, None, pa, False, return_highres=True)
                padding = None

            out = model.decode_and_predict(
                dorsal_embs.clone(), dorsal_pos, dorsal_mask, high_res_featmaps,
                ys.to(img.device), padding, ys_high.to(img.device), task_ids)
            prob, stop = out['pred_fixation_map'], out['pred_termination']
            prob = prob.view(bs, -1)
            stop_flags.append(stop)

            if pa.enforce_IOR:
                # Enforcing IOR
                batch_idx, visited_locs = torch.where(action_mask==1)
                prob[batch_idx, visited_locs] = 0

        if sample_action:
            m = Categorical(prob)
            next_word = m.sample()
        else:
            _, next_word = torch.max(prob, dim=1)
        
        next_word = next_word.cpu()
        norm_fy = (next_word // pa.im_w) / float(pa.im_h)
        norm_fx = (next_word % pa.im_w) / float(pa.im_w)
        normalized_fixs = torch.cat(
            [normalized_fixs, torch.stack([norm_fx, norm_fy], dim=1).unsqueeze(1)], dim=1)

        new_mask = get_IOR_mask(norm_fx.numpy(),
                                norm_fy.numpy(),
                                pa.im_h, 
                                pa.im_w, 
                                pa.IOR_radius)
        action_mask = torch.logical_or(action_mask, new_mask)

    stop_flags = torch.stack(stop_flags, dim=1)
    # Truncate at terminal action
    trajs = []
    fixed_eval_length = getattr(pa, "fixed_eval_length", None)
    for i in range(normalized_fixs.size(0)):
        if fixed_eval_length is not None and fixed_eval_length > 0:
            ind = min(int(fixed_eval_length), normalized_fixs.size(1))
            trajs.append(normalized_fixs[i, :ind])
            continue

        is_terminal = stop_flags[i] > 0.5
        if is_terminal.sum() == 0:
            ind = normalized_fixs.size(1)
        else:
            ind = is_terminal.to(torch.int8).argmax().item() + 1
        trajs.append(normalized_fixs[i, :ind])

    nonstop_trajs = [normalized_fixs[i] for i in range(normalized_fixs.size(0))]
    return trajs, nonstop_trajs


def actions2scanpaths(norm_fixs, patch_num, im_h, im_w):
    # convert actions to scanpaths
    scanpaths = []
    for traj in norm_fixs:
        task_name, img_name, condition, fixs = traj
        fixs = fixs.numpy()
        scanpaths.append({
            'X': fixs[:, 0] * im_w,
            'Y': fixs[:, 1] * im_h,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths

def compute_conditional_saliency_metrics(pa, model, gazeloader, task_dep_prior_maps, device):
    n_samples, info_gain, nss, auc = 0, 0, 0, 0
    for batch in tqdm(gazeloader, desc='Computing saliency metrics'):
        img = batch['true_state'].to(device)
        task_ids = batch['task_id'].to(device)
        is_last = batch['is_last']
        non_term_mask = torch.logical_not(is_last)
        if torch.sum(non_term_mask) == 0:
            continue
        # if pa.include_freeview:
        #     task_ids[batch['is_freeview']] = 18
        inp_seq, inp_seq_high = utils.transform_fixations(
            batch['normalized_fixations'], batch['is_padding'], 
            pa, False, return_highres=True)
        inp_seq = inp_seq.to(device)
        inp_padding_mask = (inp_seq == pa.pad_idx)

        gt_next_fixs = (batch['next_normalized_fixations'][:, -1] * torch.tensor(
            [pa.im_w, pa.im_h])).to(torch.long)
        prior_maps = torch.stack(
            [task_dep_prior_maps[task] for task in batch['task_name']]).cpu()
        with torch.no_grad():
            logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device), task_ids)
            pred_fix_map = logits['pred_fixation_map']
            if len(pred_fix_map.size()) > 3:
                pred_fix_map = pred_fix_map[torch.arange(img.size(0)), task_ids]
            pred_fix_map = pred_fix_map.detach().cpu()
            # pred_fix_map = torch.nn.functional.interpolate(
            #     pred_fix_map.unsqueeze(1), size=(pa.im_h, pa.im_w), mode='bilinear').squeeze(1)

            probs = pred_fix_map
            # Normalize values to 0-1
            # probs -= probs.view(probs.size(0), 1, -1).min(dim=-1, keepdim=True)[0]
            probs /= probs.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        probs = probs[non_term_mask]
        prior_maps = prior_maps[non_term_mask]
        gt_next_fixs = gt_next_fixs[non_term_mask]
        info_gain += metrics.compute_info_gain(probs, gt_next_fixs, prior_maps)
        nss += metrics.compute_NSS(probs, gt_next_fixs)
        auc += metrics.compute_cAUC(probs, gt_next_fixs)
        n_samples += gt_next_fixs.size(0)

    info_gain /= n_samples
    nss /= n_samples
    auc /= n_samples
        
    return info_gain.item(), nss.item(), auc.item()

def sample_scanpaths(model, dataloader, pa, device, sample_action, center_initial=True):

    all_actions, nonstop_actions = [], []
    for i in range(10):
        for batch in tqdm(dataloader, desc=f'Generate scanpaths [{i}/10]:'):
            img = batch['im_tensor'].to(device)
            task_ids = batch['task_id'].to(device)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            cond_batch = batch['condition']
            trajs, nonstop_trajs = scanpath_decode(
                model.module if isinstance(model, torch.nn.DataParallel) else model,
                img, task_ids, pa, sample_action, center_initial)
            nonstop_actions.extend([
                (cat_names_batch[i], img_names_batch[i],
                 cond_batch[i], nonstop_trajs[i]) for i in range(img.size(0))
            ])

            all_actions.extend([
                (cat_names_batch[i], img_names_batch[i],
                 cond_batch[i], trajs[i]) for i in range(img.size(0))
            ])

        if not sample_action:
            break
            
    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_h, pa.im_w)
    return scanpaths, nonstop_actions

def compute_multimatch_hat_fv(pred_scanpaths, gt_scanpaths, im_w, im_h, debug_max_err=8):
    """
    HAT-FV 专用 MultiMatch（只按图像名配对）
    ------------------------------------------------
    输入:
      pred_scanpaths: list[dict], 预测路径（evaluate里生成的scanpaths）
      gt_scanpaths:   list[dict], 人类GT路径（human_scanpath_test）
      im_w, im_h:     图像宽高（如512, 320）
    依赖:
      from multimatch import multimatch as mmg
    返回:
      mm_dict, per_image_results
    """
    # 防御：库不存在
    try:
        mmg  # noqa
    except NameError:
        return {
            "FV_MM_Avg": np.nan,
            "FV_MM_Avg_4D": np.nan,
            "FV_MM_Shape": np.nan,
            "FV_MM_Direction": np.nan,
            "FV_MM_Length": np.nan,
            "FV_MM_Position": np.nan,
            "FV_MM_Duration": np.nan,
            "FV_MM_valid_pairs": 0,
            "FV_MM_valid_images": 0
        }, []

    def _to_fix_array(X, Y, T):

        n = min(len(X), len(Y), len(T))

        arr = np.zeros(
            n,
            dtype=[
                ('start_x', 'f4'),
                ('start_y', 'f4'),
                ('duration', 'f4')
            ]
        )

        for i in range(n):

            arr['start_x'][i] = np.clip(float(X[i]), 0, im_w - 1)
            arr['start_y'][i] = np.clip(float(Y[i]), 0, im_h - 1)

            dur = float(T[i])

            if dur <= 0:
                dur = 1.0

            arr['duration'][i] = dur

        return arr

    # ---------- GT按name分组（FV关键：不按task） ----------
    gt_by_name = defaultdict(list)
    for g in gt_scanpaths:
        # 关键！！
        if g.get("split", None) != "valid":
            continue

        name = g.get("name", None)
        if name is None:
            continue
        X = g.get("X", [])
        Y = g.get("Y", [])
        T = g.get("T", [300.0] * len(X))
        if min(len(X), len(Y), len(T)) < 3:
            continue
        gt_by_name[name].append({
            "subject": g.get("subject", 1),
            "X": X, "Y": Y, "T": T
        })

    # ---------- 预测每图取一条（subject=1优先） ----------
    pred_by_name = {}
    for p in pred_scanpaths:
        name = p.get("name", None)
        if name is None:
            continue
        if name not in pred_by_name:
            pred_by_name[name] = p
        elif pred_by_name[name].get("subject", 999) != 1 and p.get("subject", 999) == 1:
            pred_by_name[name] = p

    per_img_vecs = []       # 每图5维均值
    per_image_results = []  # 用于可视化排序
    valid_pairs = 0
    err_cnt = 0

    common_names = set(pred_by_name.keys()) & set(gt_by_name.keys())

    if len(common_names) == 0:
        print("[WARN] FV MultiMatch: no common image names between pred and gt.")
        return {
            "FV_MM_Avg": np.nan,
            "FV_MM_Avg_4D": np.nan,
            "FV_MM_Shape": np.nan,
            "FV_MM_Direction": np.nan,
            "FV_MM_Length": np.nan,
            "FV_MM_Position": np.nan,
            "FV_MM_Duration": np.nan,
            "FV_MM_valid_pairs": 0,
            "FV_MM_valid_images": 0
        }, []

    for name in sorted(common_names):
        p = pred_by_name[name]

        pX = p.get("X", [])
        pY = p.get("Y", [])
        pT = p.get("T", [300.0] * len(pX))
        if min(len(pX), len(pY), len(pT)) < 3:
            continue

        # pred_fix = _to_fix_dict(pX, pY, pT)
        img_vecs = []

        # 与该图所有GT受试者逐一比较
        for g in gt_by_name[name]:
            gX, gY, gT = g["X"], g["Y"], g["T"]
            if min(len(gX), len(gY), len(gT)) < 3:
                continue

            # gt_fix = _to_fix_dict(gX, gY, gT)

            try:
                pred_fix = _to_fix_array(pX, pY, pT)
                gt_fix = _to_fix_array(gX, gY, gT)

                res = mmg.docomparison(
                pred_fix,
                gt_fix,
                sz=[im_w, im_h],
                grouping=False
                )

                arr = np.array(res, dtype=np.float64)

                if arr.shape == (1, 5):
                    vec = arr[0]
                else:
                    vec = None

                if vec is not None and np.all(np.isfinite(vec)):
                    img_vecs.append(vec)
                    valid_pairs += 1

            except Exception as e:
                if err_cnt < debug_max_err:
                    print(f"[MM-DEBUG-FV] exception on {name}: {repr(e)}")
                    print(f"pred_fix shape = {pred_fix.shape}")
                    print(f"gt_fix shape = {gt_fix.shape}")
                    err_cnt += 1
                continue

        if len(img_vecs) > 0:
            v = np.mean(np.stack(img_vecs, axis=0), axis=0)  # [shape, dir, len, pos, dur]
            per_img_vecs.append(v)
            per_image_results.append({
                "name": name,
                "FV_MM_Shape": float(v[0]),
                "FV_MM_Direction": float(v[1]),
                "FV_MM_Length": float(v[2]),
                "FV_MM_Position": float(v[3]),
                "FV_MM_Duration": float(v[4]),
                "FV_MM_Avg": float(np.mean(v)),
                "FV_MM_Avg_4D": float(np.mean(v[:4]))  # 主推荐，不含Duration
            })

    if len(per_img_vecs) == 0:
        print(f"[WARN] FV MultiMatch: no valid pairs after filtering. common_names={len(common_names)}")
        return {
            "FV_MM_Avg": np.nan,
            "FV_MM_Avg_4D": np.nan,
            "FV_MM_Shape": np.nan,
            "FV_MM_Direction": np.nan,
            "FV_MM_Length": np.nan,
            "FV_MM_Position": np.nan,
            "FV_MM_Duration": np.nan,
            "FV_MM_valid_pairs": 0,
            "FV_MM_valid_images": 0
        }, []

    avg_mm = np.mean(np.stack(per_img_vecs, axis=0), axis=0)
    mm_shape, mm_dir, mm_len, mm_pos, mm_dur = [float(x) for x in avg_mm]

    out = {
        "FV_MM_Avg": round(float(np.mean([mm_shape, mm_dir, mm_len, mm_pos, mm_dur])), 4),
        "FV_MM_Avg_4D": round(float(np.mean([mm_shape, mm_dir, mm_len, mm_pos])), 4),
        "FV_MM_Shape": round(mm_shape, 4),
        "FV_MM_Direction": round(mm_dir, 4),
        "FV_MM_Length": round(mm_len, 4),
        "FV_MM_Position": round(mm_pos, 4),
        "FV_MM_Duration": round(mm_dur, 4),
        "FV_MM_valid_pairs": int(valid_pairs),
        "FV_MM_valid_images": int(len(per_img_vecs))
    }

    print(f"[FV-MultiMatch] Avg5D={out['FV_MM_Avg']:.4f}, Avg4D={out['FV_MM_Avg_4D']:.4f}, "
          f"Shape={out['FV_MM_Shape']:.4f}, Dir={out['FV_MM_Direction']:.4f}, "
          f"Len={out['FV_MM_Length']:.4f}, Pos={out['FV_MM_Position']:.4f}, "
          f"Dur={out['FV_MM_Duration']:.4f}, pairs={out['FV_MM_valid_pairs']}, imgs={out['FV_MM_valid_images']}")

    return out, per_image_results

def select_multimatch_bad_cases(per_image_results,
                                metric="FV_MM_Avg_4D",
                                top_k=10):

    vals = []

    for r in per_image_results:

        if metric not in r:
            continue

        v = r[metric]

        if v is None:
            continue

        if not np.isfinite(v):
            continue

        vals.append((r["name"], float(v)))

    vals.sort(key=lambda x: x[1])  # 越低越差

    return vals[:top_k]

def pick_pred_one_per_image(pred_scanpaths):

    pred_map = {}

    for p in pred_scanpaths:

        name = p["name"]

        if name not in pred_map:
            pred_map[name] = p

    return pred_map

def build_gt_by_image(gt_scanpaths):

    gt_by_img = defaultdict(list)

    for g in gt_scanpaths:

        name = g["name"]

        gt_by_img[name].append(g)

    return gt_by_img

def visualize_multimatch_bad_cases(
        bad_cases,
        pred_scanpaths,
        gt_scanpaths,
        img_root,
        save_dir,
        W,
        H,
        metric_name="FV_MM_Avg_4D"):

    os.makedirs(save_dir, exist_ok=True)

    pred_map = pick_pred_one_per_image(pred_scanpaths)
    gt_by_img = build_gt_by_image(gt_scanpaths)

    saved = 0

    for name, score in bad_cases:

        if name not in pred_map:
            continue

        # ---------------- load image ----------------
        img_path = None

        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

            p = os.path.join(
                img_root,
                os.path.splitext(name)[0] + ext
            )

            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            continue

        bgr = cv2.imread(img_path)

        if bgr is None:
            continue

        bgr = cv2.resize(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pred = pred_map[name]
        gt_all = gt_by_img.get(name, [])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # =========================================================
        # LEFT : prediction
        # =========================================================
        ax = axes[0]

        ax.imshow(rgb)

        px = pred["X"]
        py = pred["Y"]

        ax.plot(
            px,
            py,
            "-o",
            color="red",
            linewidth=2.5,
            markersize=7,
            markerfacecolor='red',
            markeredgecolor='white'
        )

        for i, (x, y) in enumerate(zip(px, py)):

            ax.text(
                x + 4,
                y - 4,
                str(i + 1),
                color='yellow',
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='black',
                    alpha=0.6
                )
            )

        ax.set_title(
            f"Prediction ({len(px)} fixations)",
            fontsize=11,
            fontweight='bold'
        )

        ax.axis("off")

        # =========================================================
        # RIGHT : GT density + GT scanpaths
        # =========================================================
        ax = axes[1]

        ax.imshow(rgb)

        gx_all = []
        gy_all = []

        # 所有subject轨迹（淡绿色）
        for subj in gt_all:

            gx = subj["X"]
            gy = subj["Y"]

            gx_all.extend(gx)
            gy_all.extend(gy)

            ax.plot(
                gx,
                gy,
                "-o",
                color='lime',
                alpha=0.15,
                linewidth=1.0,
                markersize=3
            )

        # fixation density
        den = np.zeros((H, W), dtype=np.float32)

        for x, y in zip(gx_all, gy_all):

            ix = int(round(x))
            iy = int(round(y))

            if 0 <= ix < W and 0 <= iy < H:
                den[iy, ix] += 1

        if den.max() > 0:

            den = gaussian_filter(den, sigma=10)

            den /= den.max()

            ax.imshow(
                den,
                cmap='Greens',
                alpha=0.35
            )

        ax.scatter(
            gx_all,
            gy_all,
            c='lime',
            s=18,
            edgecolors='darkgreen',
            linewidths=0.5,
            alpha=0.5
        )

        ax.set_title(
            f"Human GT ({len(gt_all)} subjects)",
            fontsize=11,
            fontweight='bold'
        )

        ax.axis("off")

        # =========================================================
        # TITLE
        # =========================================================
        fig.suptitle(
            f"{name}\n{metric_name} = {score:.4f}  (Worst Case)",
            fontsize=13,
            fontweight='bold',
            color='darkred'
        )

        plt.tight_layout()

        out_path = os.path.join(
            save_dir,
            os.path.splitext(name)[0] + ".png"
        )

        plt.savefig(
            out_path,
            dpi=180,
            bbox_inches='tight'
        )

        plt.close(fig)

        saved += 1

    print(f"[MultiMatch] saved {saved} bad-case visualizations -> {save_dir}")

def evaluate(model,
             device,
             valid_img_loader,
             gazeloader,
             pa,
             bbox_annos,
             human_cdf,
             fix_clusters,
             task_dep_prior_maps,
             semSS_strings,
             dataset_root,
             human_scanpath_test,
             sample_action=True,
             sample_stop=False,
             output_saliency_metrics=True,
             center_initial=True,
             log_dir=None):
    print("Eval on {} batches of images and {} batches of fixations".format(
        len(valid_img_loader), len(gazeloader)))
    model.eval()
    TAP = pa.TAP
    if TAP == 'FV':
        cut1, cut2, cut3 = 4, 8, 16
    else:
        cut1, cut2, cut3 = 2, 4, 6
    
    print(f"Evaluating {TAP} with max steps to be {pa.max_traj_length} " +
          f"with initial center fixation = {center_initial} " + 
          f"and enforce IOR = {pa.enforce_IOR} with radius {pa.IOR_radius}...")
    scanpaths, nonstop_actions = sample_scanpaths(
        model, valid_img_loader, pa, device, sample_action, center_initial)

    # if sample_action:
    #     nonstop_scanpaths = scanpaths
    # else:
    nonstop_scanpaths = actions2scanpaths(nonstop_actions, pa.patch_num, pa.im_h, pa.im_w)

    print('Computing metrics...')
    metrics_dict = {}
    if TAP == 'TP':
        if not sample_stop:
            utils.cutFixOnTarget(scanpaths, bbox_annos)
        # search effiency
        mean_cdf, _ = utils.compute_search_cdf(
            scanpaths, bbox_annos, pa.max_traj_length)
        metrics_dict.update(dict(zip([f"TFP_top{i}" for i in range(
                1, len(mean_cdf))], mean_cdf[1:])))

        # probability mismatch
        metrics_dict['prob_mismatch'] = np.sum(np.abs(human_cdf[:len(mean_cdf)] - mean_cdf))

    # sequence score
    def safe_seq_score(preds, clusters, max_step, truncate_gt):
        try:
            return metrics.get_seq_score(preds, clusters, max_step, truncate_gt)
        except Exception as e:
            print(f"[warn] Sequence Score skipped: {e}")
            return None

    ss_2steps = safe_seq_score(nonstop_scanpaths, fix_clusters, cut1, True)
    ss_4steps = safe_seq_score(nonstop_scanpaths, fix_clusters, cut2, True)
    ss_6steps = safe_seq_score(nonstop_scanpaths, fix_clusters, cut3, True)
    ss = safe_seq_score(scanpaths, fix_clusters, pa.max_traj_length, False)

    metrics_dict.update({
        f"{TAP}_seq_score_max": ss,
        f"{TAP}_seq_score_{cut1}steps": ss_2steps,
        f"{TAP}_seq_score_{cut2}steps": ss_4steps,
        f"{TAP}_seq_score_{cut3}steps": ss_6steps,
    })

    if TAP == 'FV':
        mm_dict, per_img_mm = compute_multimatch_hat_fv(
            scanpaths, human_scanpath_test, pa.im_w, pa.im_h
        )
        # 如果你这里有 Greedy_ 前缀逻辑，先不加前缀，后面统一加
        metrics_dict.update(mm_dict)

    if semSS_strings is not None and TAP != 'FV':
        sss_2steps = metrics.get_semantic_seq_score(
            nonstop_scanpaths, semSS_strings, cut1, 
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
        sss_4steps = metrics.get_semantic_seq_score(
            nonstop_scanpaths, semSS_strings, cut2, 
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
        sss_6steps = metrics.get_semantic_seq_score(
            nonstop_scanpaths, semSS_strings, cut3, 
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
        sss = metrics.get_semantic_seq_score(
            scanpaths, semSS_strings, pa.max_traj_length, 
            f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)
        metrics_dict.update({
            f"{TAP}_semantic_seq_score_max": sss,
            f"{TAP}_semantic_seq_score_{cut1}steps": sss_2steps,
            f"{TAP}_semantic_seq_score_{cut2}steps": sss_4steps,
            f"{TAP}_semantic_seq_score_{cut3}steps": sss_6steps,
        })

    if output_saliency_metrics:
        # temporal spatial saliency metrics
        ig, nss, auc = compute_conditional_saliency_metrics(
            pa, model, gazeloader, task_dep_prior_maps, device)
        metrics_dict.update({
            f"{TAP}_cIG": ig,
            f"{TAP}_cNSS": nss,
            f"{TAP}_cAUC": auc,
        })

    sp_len_diff = []
    # for traj in scanpaths:
    #     gt_trajs = list(
    #         filter(lambda x: x['task'] == traj['task'] and x['name'] == traj['name'],
    #                human_scanpath_test))
    #     sp_len_diff.append(len(traj['X']) - np.array([len(traj['X']) for traj in gt_trajs]))
    # sp_len_diff = np.abs(np.concatenate(sp_len_diff))
    # metrics_dict[f'{TAP}_sp_len_err_mean'] = sp_len_diff.mean()
    # metrics_dict[f'{TAP}_sp_len_err_std'] = sp_len_diff.std()
    # metrics_dict[f'{TAP}_avg_sp_len'] = np.mean([len(x['X']) for x in scanpaths])
    
    for traj in scanpaths:
        gt_trajs = [x for x in human_scanpath_test
                if x['name'] == traj['name'] and
                (TAP == 'FV' or x.get('task', 'none') == traj.get('task', 'none'))]

        if len(gt_trajs) == 0:
            continue

        gt_lens = np.array([len(x['X']) for x in gt_trajs], dtype=np.float32)
        pred_len = len(traj['X'])

        # 你可以选平均误差 or 最小误差，这里用平均误差
        diffs = np.abs(pred_len - gt_lens)
        sp_len_diff.extend(diffs.tolist())

    if len(sp_len_diff) == 0:
        metrics_dict[f'{TAP}_sp_len_err_mean'] = None
        metrics_dict[f'{TAP}_sp_len_err_std'] = None
    else:
        sp_len_diff = np.array(sp_len_diff, dtype=np.float32)
        metrics_dict[f'{TAP}_sp_len_err_mean'] = float(sp_len_diff.mean())
        metrics_dict[f'{TAP}_sp_len_err_std'] = float(sp_len_diff.std())
    metrics_dict[f'{TAP}_avg_sp_len'] = float(np.mean([len(x['X']) for x in scanpaths]))

    if not sample_action:
        prefix = 'Greedy_'
        keys = list(metrics_dict.keys())
        for k in keys:
            metrics_dict[prefix + k] = metrics_dict.pop(k)

    if TAP == 'FV' and len(per_img_mm) > 0:

        bad_cases = select_multimatch_bad_cases(
            per_img_mm,
            metric="FV_MM_Avg_4D",
            top_k=10
        )

        visualize_multimatch_bad_cases(
            bad_cases,
            scanpaths,
            human_scanpath_test,
            img_root=f"{dataset_root}/images",
            save_dir=join(log_dir, "FV_MM_Worst10"),
            W=pa.im_w,
            H=pa.im_h,
            metric_name="FV_MM_Avg_4D"
        )
    if log_dir is not None:
        for sp in scanpaths:
            sp['X'] = sp['X'].tolist()
            sp['Y'] = sp['Y'].tolist()
        with open(join(log_dir, f'predictions_{TAP}.json'), 'w') as f:
            json.dump(scanpaths, f, indent=4)
        with open(join(log_dir, f'metrics_{TAP}.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4, cls=NumpyEncoder)
    return metrics_dict, scanpaths
