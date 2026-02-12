import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# from .relobralo import ReLoBRaLo

from .utils_losses import plate_stress_loss, bc_edgeY_loss, bc_edgeX_loss

# plotting function
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=5)
    # Add a colorbar
    plt.colorbar(label='f')

# =============================================================================
# 1. Validation Function (验证集)
# =============================================================================
def val(model, loader, args, device, node_counts):
    # 索引计算
    max_pde = node_counts['n_pde']
    max_load = node_counts['n_load']
    max_free = node_counts['n_free']
    max_fix = node_counts['n_fix']
    max_hole = node_counts['n_hole']

    # 计算各部分在 coor_field 中的索引范围 (Order: PDE -> Load -> Free -> Fix -> Hole)
    idx_load_start = max_pde
    idx_load_end = idx_load_start + max_load

    idx_free_start = idx_load_end
    idx_free_end = idx_free_start + max_free

    idx_fix_start = idx_free_end
    idx_fix_end = idx_fix_start + max_fix

    idx_hole_start = idx_fix_end
    idx_hole_end = idx_hole_start + max_hole

    mean_relative_L2 = 0
    num_eval = 0

    for batch_idx, data in enumerate(loader):
        # 1. 解包 9 个变量
        b_force = data[0].to(device)
        b_disp = data[1].to(device)
        b_E = data[2].to(device)
        b_nu = data[3].to(device)
        b_geo = data[4].to(device)

        coor_field = data[5].to(device)  # (B, N, 2) 包含全域点
        u_true = data[6].to(device)
        v_true = data[7].to(device)
        flag = data[8].to(device)

        # 2. [Fix 2] 几何编码切片 (Geometry Slicing)
        # 严格遵循 PI-GANO 的几何提取策略
        if args.geo_node == 'vary_bound' or args.geo_node == 'vary_bound_sup':
            ss_index = np.arange(idx_hole_start, idx_hole_end)
        elif args.geo_node == 'all_bound':
            ss_index = np.arange(idx_load_start, idx_hole_end)
        elif args.geo_node == 'all_domain':
            ss_index = np.arange(0, idx_hole_end)
        else:
            raise ValueError(
                f"Invalid args.geo_node: {args.geo_node}. Expected 'vary_bound', 'all_bound', or 'all_domain'.")

        # 提取几何点云
        shape_coor = coor_field[:, ss_index, :]  # (B, M_geo, 2)
        shape_flag = flag[:, ss_index]  # (B, M_geo)

        # 3. [Fix 3] 全域坐标提取 (用于 Trunk 输入)
        # 直接使用 coor_field 的全场坐标进行预测
        x_full = coor_field[..., 0]
        y_full = coor_field[..., 1]

        # 4. 模型前向
        u_pred, v_pred = model(
            b_force, b_disp, b_E, b_nu, b_geo,
            x_full, y_full,  # Trunk 输入: 全域坐标
            shape_coor, shape_flag  # DG 输入: 切片后的几何点
        )

        # 5. 全域误差计算
        # flag > 0.5 会自动过滤掉 Padding (0)，保留 PDE+边界所有点
        mask = (flag > 0.5).float()

        diff_sq = ((u_pred - u_true) * mask) ** 2 + ((v_pred - v_true) * mask) ** 2
        gt_sq = (u_true * mask) ** 2 + (v_true * mask) ** 2

        # 相对 L2 误差公式
        numerator = torch.sqrt(torch.sum(diff_sq, dim=1))
        denominator = torch.sqrt(torch.sum(gt_sq, dim=1))

        L2_relative = numerator / (denominator + 1e-8)

        mean_relative_L2 += torch.sum(L2_relative).item()
        num_eval += b_force.shape[0]

    return mean_relative_L2 / num_eval

# =============================================================================
# 2. Testing Function (测试集 - 修正几何编码与全域坐标提取)
# =============================================================================
def test(model, loader, args, device, node_counts, dir):

    model.eval()

    # [Fix 1] 同样的索引计算逻辑
    max_pde = node_counts['n_pde']
    max_load = node_counts['n_load']
    max_free = node_counts['n_free']
    max_fix = node_counts['n_fix']
    max_hole = node_counts['n_hole']

    idx_load_start = max_pde
    idx_hole_start = max_pde + max_load + max_free + max_fix
    idx_hole_end = idx_hole_start + max_hole
    # (其他中间索引如果只用于 geo_node 判断，可以用上面的边界推导)

    mean_relative_L2 = 0
    num_eval = 0
    max_relative_err = -1
    min_relative_err = np.inf

    for batch_idx, data in enumerate(loader):

        b_force = data[0].to(device)
        b_disp = data[1].to(device)
        b_E = data[2].to(device)
        b_nu = data[3].to(device)
        b_geo = data[4].to(device)

        coor_field = data[5].to(device)
        u_true = data[6].to(device)
        v_true = data[7].to(device)
        flag = data[8].to(device)

        # [Fix 2] 几何编码切片
        if args.geo_node == 'vary_bound' or args.geo_node == 'vary_bound_sup':
            ss_index = np.arange(idx_hole_start, idx_hole_end)
        elif args.geo_node == 'all_bound':
            # Load 起始点到 Hole 结束点
            ss_index = np.arange(idx_load_start, idx_hole_end)
        elif args.geo_node == 'all_domain':
            ss_index = np.arange(0, idx_hole_end)
        else:
            raise ValueError(
                f"Invalid args.geo_node: {args.geo_node}. Expected 'vary_bound', 'all_bound', or 'all_domain'.")

        shape_coor = coor_field[:, ss_index, :]
        shape_flag = flag[:, ss_index]

        # [Fix 3] 显式提取测试集坐标 (全域)
        # coor_field 包含了 (Batch, N_all_nodes, 2)
        x_full = coor_field[..., 0]
        y_full = coor_field[..., 1]

        # 模型前向
        u_pred, v_pred = model(
            b_force, b_disp, b_E, b_nu, b_geo,
            x_full, y_full,
            shape_coor, shape_flag
        )

        # 计算误差
        mask = (flag > 0.5).float()
        diff_sq = ((u_pred - u_true) * mask) ** 2 + ((v_pred - v_true) * mask) ** 2
        gt_sq = (u_true * mask) ** 2 + (v_true * mask) ** 2

        L2_relative = torch.sqrt(torch.sum(diff_sq, 1)) / (torch.sqrt(torch.sum(gt_sq, 1)) + 1e-8)

        # --- 数据收集用于绘图 (保持不变，仅适配变量名) ---
        if dir == 'x':
            pred, gt = u_pred, u_true
        if dir == 'y':
            pred, gt = v_pred, v_true

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_xcoor = coor_field[max_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_ycoor = coor_field[max_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_f = pred[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_gt = gt[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_ff = flag[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(worst_ff>0.5)[0]
            worst_xcoor = worst_xcoor[valid_id]
            worst_ycoor = worst_ycoor[valid_id]
            worst_f = worst_f[valid_id]
            worst_gt = worst_gt[valid_id]
            worst_ff = worst_ff[valid_id]
        min_err, min_err_idx = torch.topk(-L2_relative, 1)
        min_err = -min_err
        if min_err < min_relative_err:
            min_relative_err = min_err
            best_xcoor = coor_field[min_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_ycoor = coor_field[min_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_f = pred[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            best_gt = gt[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            best_ff = flag[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(best_ff>=0.5)[0]
            best_xcoor = best_xcoor[valid_id]
            best_ycoor = best_ycoor[valid_id]
            best_f = best_f[valid_id]
            best_gt = best_gt[valid_id]
            best_ff = best_ff[valid_id]

        mean_relative_L2 += torch.sum(L2_relative).item()
        num_eval += b_force.shape[0]

    # color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # errorcolor bar range
    err_max_color = np.amax([np.amax(np.abs(worst_f - worst_gt)), np.amax(np.abs(best_f - best_gt))])

    # make the plot
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2, 3, 2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f - worst_gt), cmap=cm, vmin=0, vmax=err_max_color, marker='o',
                s=3)
    plt.title('absolute error')
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2, 3, 5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f - best_gt), cmap=cm, vmin=0, vmax=err_max_color, marker='o',
                s=3)
    plt.title('absolute error')
    plt.colorbar()
    plt.savefig(r'./res/plots/sample_{}_{}_{}_{}.png'.format(args.geo_node, args.model, args.data, dir))

    return mean_relative_L2 / num_eval


# function of extracting the geometry embeddings
def get_geometry_embeddings(model, loader, args, device, node_counts):
    # transforme state to be eval
    model.eval()

    # [Fix 1] 解包字典格式的节点计数
    max_pde = node_counts['n_pde']
    max_load = node_counts['n_load']
    max_free = node_counts['n_free']
    max_fix = node_counts['n_fix']
    max_hole = node_counts['n_hole']

    # [Fix 2] 预计算索引 (Order: PDE -> Load -> Free -> Fix -> Hole)
    idx_load_start = max_pde
    idx_hole_start = max_pde + max_load + max_free + max_fix
    idx_hole_end = idx_hole_start + max_hole

    # forward to get the embeddings
    all_geo_embeddings = []

    # [Fix 3] 循环解包 9 个变量
    for batch_idx, data in enumerate(loader):
        # 虽然我们只需要坐标和Flag，但为了格式统一，建议完整解包
        # b_force = data[0].to(device) ... (不需要)

        coor_field = data[5].to(device)
        flag = data[8].to(device)

        # [Fix 4] 几何编码切片 (Geometry Slicing)
        # 逻辑与 train/val/test 保持高度一致
        if args.geo_node == 'vary_bound' or args.geo_node == 'vary_bound_sup':
            ss_index = np.arange(idx_hole_start, idx_hole_end)
        elif args.geo_node == 'all_bound':
            ss_index = np.arange(idx_load_start, idx_hole_end)  # Load -> Hole
        elif args.geo_node == 'all_domain':
            ss_index = np.arange(0, idx_hole_end)
        else:
            raise ValueError(
                f"Invalid args.geo_node: {args.geo_node}. Expected 'vary_bound', 'all_bound', or 'all_domain'.")

        # 提取几何点云
        shape_coors = coor_field[:, ss_index, :]  # (B, M_geo, 2)
        shape_flag = flag[:, ss_index]  # (B, M_geo)

        # [Fix 5] 模型前向
        # 直接调用模型的 DG 子模块，它只需要几何输入
        with torch.no_grad():
            # DG forward 返回的是 Domain_enc (B, 1, F)
            # 我们通常需要 squeeze 掉中间的维度变成 (B, F) 以便分析
            geo_emb = model.DG(shape_coors, shape_flag).squeeze(1)

        all_geo_embeddings.append(geo_emb)

    all_geo_embeddings = torch.cat(tuple(all_geo_embeddings), 0)

    return all_geo_embeddings


# =============================================================================
# 3. Training Function
# =============================================================================
def train(args, config, model, device, loaders, node_counts):
    # --- 初始化权重打印 ---
    print('Training configuration...')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])
    debug_print = bool(config['train'].get('debug_print', False))

    train_loader, val_loader, test_loader = loaders

    # 解包节点数
    max_pde = node_counts['n_pde']
    max_load = node_counts['n_load']
    max_free = node_counts['n_free']
    max_fix = node_counts['n_fix']
    max_hole = node_counts['n_hole']

    # --- 优化器与Loss ---
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])
    mse = nn.MSELoss()

    # 可视化频率
    vf = config['train']['visual_freq']
    err_hist = []

    # 加载预训练模型 (如果有)
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))
        print("Loaded existing best model.")
    except:
        print('No trained models, starting from scratch.')

    model = model.to(device)

    # --- 预计算索引范围 ---
    # 索引顺序: PDE -> Load -> Free -> Fix -> Hole
    idx_load_start = max_pde
    idx_load_end = idx_load_start + max_load

    idx_free_start = idx_load_end
    idx_free_end = idx_free_start + max_free

    idx_fix_start = idx_free_end
    idx_fix_end = idx_fix_start + max_fix

    idx_hole_start = idx_fix_end
    idx_hole_end = idx_hole_start + max_hole

    # --- 初始化 ReLoBRaLo 动态权重 ---
    # 关注 4 项 Loss: 0:PDE, 1:Load, 2:Free, 3:Fix
    # relobralo = ReLoBRaLo(num_losses=4, alpha=0.999, temperature=0.1, rho=0.999, device=device)
    # print("Initializing ReLoBRaLo for dynamic loss weighting...")

    # [新增] 定义固定权重 (你可以手动调节这些值)
    w_pde = 1.0
    w_load = 10
    w_free = 1.0
    w_fix = 10

    # 训练主循环
    pbar = tqdm(range(config['train']['epochs']), dynamic_ncols=True, smoothing=0.1)
    min_val_err = np.inf
    avg_pde_loss = 0
    avg_load_loss = 0
    avg_free_loss = 0
    avg_fix_loss = 0

    # 记录当前权重以便打印
    current_weights = torch.ones(4)

    for e in pbar:
        # --- Validation & Printing Block (验证与打印模块) ---
        if e % vf == 0:
            model.eval()
            with torch.no_grad():
                val_err = val(model, val_loader, args, device, node_counts)

                err_hist.append(val_err)

                # print(f'\nEpoch {e}:')
                print('Current epoch error:', val_err)
                print('current epochs pde loss:', avg_pde_loss)
                print('fix bc loss:', avg_fix_loss)
                print('free bc loss:', avg_free_loss)
                print('load bc loss:', avg_load_loss)
                # print(f'Current Weights [PDE, Load, Fix, Free]: {current_weights.cpu().numpy()}')

                avg_pde_loss = 0
                avg_fix_loss = 0
                avg_free_loss = 0
                avg_load_loss = 0

            if val_err < min_val_err:
                min_val_err = val_err
                torch.save(model.state_dict(),
                           r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model))

        model.train()

        for batch_idx, data in enumerate(train_loader):

            for _ in range(config['train']['coor_sampling_freq']):

                # ================= 1. 数据解包 (适配 9 参数) =================
                b_force = data[0].to(device)
                b_disp = data[1].to(device)
                b_E = data[2].to(device)
                b_nu = data[3].to(device)
                b_geo = data[4].to(device)

                coor_field = data[5].to(device)
                u_true = data[6].to(device)
                v_true = data[7].to(device)
                flag = data[8].to(device)

                # ================= 2. 坐标准备 (开启梯度) =================
                # 必须开启梯度，因为 PDE 和 Free 边界需要求导
                x_all = coor_field[..., 0]
                y_all = coor_field[..., 1]
                x_all.requires_grad_(True)
                y_all.requires_grad_(True)

                # ================= 3. 数据切片 (Slicing) =================

                # (A) PDE 内部点 (随机采样)
                ss_idx_pde = np.random.choice(np.arange(max_pde), config['train']['coor_sampling_size'])
                x_pde = x_all[:, ss_idx_pde]
                y_pde = y_all[:, ss_idx_pde]
                pde_flag = flag[:, ss_idx_pde]

                # (B) Load Boundary (受载边界)
                ss_idx_load = np.arange(idx_load_start, idx_load_end)
                x_load = x_all[:, ss_idx_load]
                y_load = y_all[:, ss_idx_load]
                u_load_gt = u_true[:, ss_idx_load]
                v_load_gt = v_true[:, ss_idx_load]
                load_flag = flag[:, ss_idx_load]

                # (C) Free Boundary (自由边界)
                ss_idx_free = np.arange(idx_free_start, idx_free_end)
                x_free = x_all[:, ss_idx_free]
                y_free = y_all[:, ss_idx_free]
                free_flag = flag[:, ss_idx_free]

                # (D) Fix Boundary (固定边 + 孔边) -> 都是位移约束
                ss_idx_bcxy = np.concatenate([
                    np.arange(idx_fix_start, idx_fix_end),
                    np.arange(idx_hole_start, idx_hole_end)
                ])
                x_fix = x_all[:, ss_idx_bcxy]
                y_fix = y_all[:, ss_idx_bcxy]
                fix_flag = flag[:, ss_idx_bcxy]

                # (E) Geometry Inputs (修复: 根据 args.geo_node 提取几何特征点)
                if args.geo_node == 'vary_bound' or args.geo_node == 'vary_bound_sup':
                    ss_idx_geo = np.arange(idx_hole_start, idx_hole_end)
                elif args.geo_node == 'all_bound':
                    ss_idx_geo = np.arange(idx_load_start, idx_hole_end)
                elif args.geo_node == 'all_domain':
                    ss_idx_geo = np.arange(0, idx_hole_end)
                else:
                    raise ValueError(
                        f"Invalid args.geo_node: {args.geo_node}. Expected 'vary_bound', 'all_bound', or 'all_domain'.")

                shape_coor = coor_field[:, ss_idx_geo, :]
                shape_flag = flag[:, ss_idx_geo]

                # ================= 4. 模型前向 (Forward) =================

                # 1. PDE 预测
                u_pde_pred, v_pde_pred = model(b_force, b_disp, b_E, b_nu, b_geo, x_pde, y_pde, shape_coor, shape_flag)

                # 2. Load 预测
                u_load_pred, v_load_pred = model(b_force, b_disp, b_E, b_nu, b_geo, x_load, y_load, shape_coor, shape_flag)

                # 3. Free 预测
                u_free_pred, v_free_pred = model(b_force, b_disp, b_E, b_nu, b_geo, x_free, y_free, shape_coor, shape_flag)

                # 4. Fix 预测
                u_fix_pred, v_fix_pred = model(b_force, b_disp, b_E, b_nu, b_geo, x_fix, y_fix, shape_coor, shape_flag)

                # =========================================================
                # [插入] 调试探针：检查进入 Loss 计算前的 PDE 点状态
                # =========================================================
                if debug_print and batch_idx == 0 and e == 0:  # 只在第一个 Epoch 的第一个 Batch 打印一次，防止刷屏
                    print(f"\n======== DEBUG: Inside Training Loop (Before Loss) ========")
                    print(f"1. PDE Coordinates (x_pde):")
                    print(f"   - Shape: {x_pde.shape}")
                    print(f"   - Requires Grad: {x_pde.requires_grad} (Must be True!)")
                    print(f"   - Min / Max Value: {x_pde.min().item():.4f} / {x_pde.max().item():.4f}")

                    print(f"2. PDE Predictions (u_pde_pred):")
                    print(f"   - Shape: {u_pde_pred.shape}")
                    print(f"   - Min / Max Value: {u_pde_pred.min().item():.4f} / {u_pde_pred.max().item():.4f}")
                    print(f"   - Is all zero? {torch.all(u_pde_pred == 0).item()}")

                    print(f"3. Sampling Logic:")
                    print(f"   - Max PDE Index: {max_pde}")
                    print(f"   - Sampled Indices (first 10): {ss_idx_pde[:10]}")
                    print("===========================================================\n")

                    # =========================================================
                    # [全方位调试探针] 检查所有类型的点
                    # =========================================================
                    if debug_print and batch_idx == 0 and e == 0:
                        print(f"\n======== DEBUG: Data & Prediction Inspection (Epoch 0, Batch 0) ========")

                        def inspect_tensor(name, tensor):
                            min_val = tensor.min().item()
                            max_val = tensor.max().item()
                            mean_val = tensor.mean().item()
                            print(f"  > {name}:")
                            print(f"    Shape: {tensor.shape}")
                            # 使用科学计数法 .4e 彻底区分 0.0 和 1e-12
                            print(f"    Range: [{min_val:.4e}, {max_val:.4e}] | Mean: {mean_val:.4e}")
                            if tensor.requires_grad:
                                print(f"    Requires Grad: YES")
                            else:
                                print(f"    Requires Grad: NO (Check if this is input or GT)")

                        # 1. 检查 PDE (内部点)
                        print("\n[1. PDE Points]")
                        inspect_tensor("X Coordinate", x_pde)
                        inspect_tensor("U Prediction", u_pde_pred)

                        # 2. 检查 Load (受力边界)
                        print("\n[2. Load Boundary]")
                        inspect_tensor("X Coordinate", x_load)
                        inspect_tensor("U Prediction", u_load_pred)
                        inspect_tensor("U GroundTruth", u_load_gt)  # 看看真值到底有多小

                        # 3. 检查 Free (自由边界)
                        print("\n[3. Free Boundary]")
                        inspect_tensor("X Coordinate", x_free)
                        inspect_tensor("U Prediction", u_free_pred)

                        # 4. 检查 Fix (固定边界)
                        print("\n[4. Fix Boundary]")
                        inspect_tensor("X Coordinate", x_fix)
                        inspect_tensor("U Prediction", u_fix_pred)

                        print("\n========================================================================\n")

                # ================= 5. Loss 计算 (Hybrid Strategy) =================
                #
                # [Loss 1] PDE Loss (物理驱动)
                # 传入 b_E, b_nu 用于计算应力
                rx, ry = plate_stress_loss(u_pde_pred, v_pde_pred, x_pde, y_pde, b_E, b_nu)
                loss_pde = torch.mean((rx * pde_flag) ** 2) + torch.mean((ry * pde_flag) ** 2)

                # [Loss 2] Load Loss (数据驱动)
                # 直接计算位移 MSE
                loss_load = mse(u_load_pred * load_flag, u_load_gt * load_flag) + \
                            mse(v_load_pred * load_flag, v_load_gt * load_flag)

                # [Loss 3] Free Loss (物理驱动)
                # 自由边界应力为0。假设是左右边界(EdgeX)，如果含上下边界需加 bc_edgeY_loss
                sig_xx, sig_xy = bc_edgeX_loss(u_free_pred, v_free_pred, x_free, y_free, b_E, b_nu)
                loss_free = torch.mean((sig_xx * free_flag) ** 2) + torch.mean((sig_xy * free_flag) ** 2)

                # [Loss 4] Fix Loss (数据驱动)
                loss_fix = torch.mean((u_fix_pred * fix_flag) ** 2) + \
                           torch.mean((v_fix_pred * fix_flag) ** 2)

                # ================= 6. 反向传播 =================

                # # 收集所有 Loss
                # loss_list = torch.stack([loss_pde, loss_load, loss_free, loss_fix])
                #
                # # ReLoBRaLo 计算动态权重
                # weights = relobralo.update(loss_list.detach())
                # current_weights = weights.detach()

                # 加权求和
                total_loss = w_pde * loss_pde + w_load * loss_load + w_free * loss_free + w_fix * loss_fix

                # 累加记录
                avg_pde_loss += loss_pde.item()
                avg_load_loss += loss_load.item()
                avg_free_loss += loss_free.item()
                avg_fix_loss += loss_fix.item()

                # 更新参数
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # clear cuda
                torch.cuda.empty_cache()



    # Final Test
    print("Optimization finished. Running final test...")
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))
    model.eval()

    # 1. 测试 X 方向并绘图
    print("Testing X direction...")
    err_x = test(model, test_loader, args, device, node_counts, dir='x')

    # 2. [新增] 测试 Y 方向并绘图
    print("Testing Y direction...")
    err_y = test(model, test_loader, args, device, node_counts, dir='y')

    print(f'Final Best L2 relative error (X): {err_x:.6f}')
    print(f'Final Best L2 relative error (Y): {err_y:.6f}')

