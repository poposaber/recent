#!/usr/bin/env python3
"""
簡單工具：讀取 .h5 檔案，針對指定維度畫出 real / fake 的 density distribution。

用法範例：
python plot_h5_distribution.py --file data.h5 --dim 5
python plot_h5_distribution.py --file data.h5 --dim 5 --output out.png

說明：
- 程式會自動搜尋 h5 內名稱含 'real' / 'fake' 的 dataset（不區分大小寫）
- 對指定維度 --dim N，直接提取該維的數值
- 畫出 real/fake 兩組資料在該維度上的 density distribution（不同顏色）
- 若無法自動找到 real/fake datasets，可先用 `--list-datasets` 查看 h5 的結構
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def find_real_fake_datasets(h5f):
    real = []
    fake = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            lname = name.lower()
            if 'real' in lname:
                real.append(name)
            if 'fake' in lname:
                fake.append(name)

    h5f.visititems(visitor)
    return real, fake


def aggregate_values(h5f, paths, dim_idx):
    """讀取所有 datasets，提取指定維度的數值"""
    vals = []
    for p in paths:
        try:
            ds = h5f[p]
        except Exception:
            continue
        try:
            arr = ds[()]  # load to memory
            arr = np.array(arr)
            
            if arr.size == 0:
                continue
            
            # 提取指定維度的數值
            if arr.ndim == 1:
                if dim_idx != 0:
                    raise ValueError(f'維度索引 {dim_idx} 超出 1D dataset 的範圍')
                v = arr
            else:
                # 假設 shape 為 (samples, features)，提取 features 中的第 dim_idx 個
                if dim_idx >= arr.shape[-1]:
                    raise ValueError(f'維度索引 {dim_idx} 超出 dataset 最後一軸大小 {arr.shape[-1]} 的範圍')
                v = arr[..., dim_idx]  # 提取該維度的所有樣本值
            
            if v.size:
                vals.append(v)
        except Exception as e:
            print(f'跳過 {p}: {e}', file=sys.stderr)
            continue

    if not vals:
        return np.array([])
    return np.concatenate(vals, axis=0)


def main():
    p = argparse.ArgumentParser(description='Plot real/fake density from .h5')
    p.add_argument('--file', '-f', required=True, help='.h5 檔案路徑')
    p.add_argument('--dim', '-d', type=int, required=True, help='要提取的維度索引（從0開始）')
    p.add_argument('--output', '-o', help='輸出圖檔 (png)。若不指定則顯示視窗。')
    p.add_argument('--bins', type=int, default=100, help='當 seaborn 不可用時 histogram 的 bins 數量')
    p.add_argument('--list-datasets', action='store_true', help='只列出 h5 內的 datasets 並離開')

    args = p.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        print('找不到檔案：', args.file, file=sys.stderr)
        sys.exit(2)

    with h5py.File(str(fpath), 'r') as h5f:
        # list all datasets
        all_dsets = []
        def _v(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_dsets.append(name)

        h5f.visititems(_v)

        if args.list_datasets:
            print('Datasets:')
            for d in all_dsets:
                print(' -', d)
            return

        real_paths, fake_paths = find_real_fake_datasets(h5f)

        if not real_paths and not fake_paths:
            print('在 h5 找不到名稱含 real 或 fake 的 dataset。', file=sys.stderr)
            print('已列出所有 datasets：', file=sys.stderr)
            for d in all_dsets:
                print(' -', d, file=sys.stderr)
            print('\n提示：請檢查 h5 結構並確保 real/fake datasets 的名稱含有「real」或「fake」關鍵字。', file=sys.stderr)
            sys.exit(1)

        print('找到的 real datasets:', real_paths)
        print('找到的 fake datasets:', fake_paths)

        real_vals = aggregate_values(h5f, real_paths, args.dim)
        fake_vals = aggregate_values(h5f, fake_paths, args.dim)

        if real_vals.size == 0 and fake_vals.size == 0:
            print('未能從指定 datasets 擷取數值。', file=sys.stderr)
            sys.exit(1)

        plt.figure(figsize=(8, 5))
        if _HAS_SEABORN:
            if real_vals.size:
                sns.kdeplot(real_vals, label='real', color='C0', fill=True)
            if fake_vals.size:
                sns.kdeplot(fake_vals, label='fake', color='C1', fill=True)
        else:
            if real_vals.size:
                plt.hist(real_vals, bins=args.bins, density=True, alpha=0.5, color='C0', label='real')
            if fake_vals.size:
                plt.hist(fake_vals, bins=args.bins, density=True, alpha=0.5, color='C1', label='fake')

        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Dimension {args.dim} Density Distribution: {fpath.name}')
        plt.legend()
        plt.tight_layout()

        if args.output:
            outp = Path(args.output)
            plt.savefig(str(outp), dpi=200)
            print('已儲存圖檔：', outp)
        else:
            plt.show()


if __name__ == '__main__':
    main()
