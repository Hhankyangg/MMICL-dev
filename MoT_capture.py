import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 避免GUI错误
width = 1000 # 图像宽度
height = 1000 # 图像高度
dpi = 100 # 图像分辨率

# 修改rc参数
plt.rcParams['figure.figsize'] = [width/float(dpi), height/float(dpi)]
plt.rcParams['figure.dpi'] = dpi

# 绘制图像
fig, ax = plt.subplots()
ax.plot([1,2,3],[4,5,6])
plt.show()
class MOTAttentionCapture:
    def __init__(self):
        self.attentions = {}  # 存储所有注意力数据
        self.hooks = []
        self.counter=0
        self.signal=True
        self.prefix='language_model.model.layers.'
        self.qfix='.self_attn.q_proj'
        self.kfix='.self_attn.k_proj'
    def _get_qk_hook(self, layer_path: str):
        """获取QK注意力钩子"""
        def hook(module, input, output):
            # 存储原始输入和输出
            self.counter+=1
            if input[0].detach().cpu().shape[0]!=2 and self.signal==True:
                if layer_path in list(self.attentions.keys()):
                    self.attentions[layer_path] = {
                        'input': torch.cat([input[0].detach().cpu(),self.attentions[layer_path]['input']],dim=0) if input[0] is not None else self.attentions[layer_path]['input'],
                        'output': torch.cat([output.detach().cpu(),self.attentions[layer_path]['output']],dim=0),
                        'module_type': type(module).__name__
                    }
                else:
                    self.attentions[layer_path] = {
                        'input': input[0].detach().cpu(),
                        'output': output.detach().cpu(),
                        'module_type': type(module).__name__
                    }
                # if '20' in layer_path and 'moe_gen' not in layer_path and 'q_proj' in layer_path:
                #     print(input[0].detach().cpu().shape)
        return hook
    
    def attach_hooks(self, model):
        """为模型附加钩子
        
        Args:
            model: 要监控的模型
            layer_names: 要监控的层名称列表，如果为None则自动查找所有QK层
        """
        self.hooks = []
        # 自动查找所有QK层
        for name, module in model.named_modules():
            if self._is_q_layer(name) or self._is_k_layer(name):
                # print(name)
                hook = module.register_forward_hook(
                    self._get_qk_hook(name)
                )
                self.hooks.append(hook)
                    
        print(f"总共监控了 {len(self.hooks)} 个QK层")
    
    def _is_q_layer(self, name):
        return ('q_proj' in name and 'language_model' in name and 'moe' not in name)

    def _is_k_layer(self, name):
        return ('k_proj' in name and 'language_model' in name and 'moe' not in name)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_data(self, clear=True):
        """获取注意力数据"""
        data = self.attentions.copy()
        if clear:
            self.attentions.clear()
        return data
    
    def compute_batch_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        aidx,
        bidx,
        scale: float = None,
        normalize: bool = True
    ):
        Q=Q.reshape(Q.shape[0],-1,128)
        K=K.reshape(K.shape[0],-1,128)
        num_heads_q = Q.shape[1]
        num_heads_k = K.shape[1]
        head_dim = Q.shape[2]
        
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)
        len_a=len(aidx)
        len_b=K.shape[0]
        Q_batch = Q[aidx,:,:].unsqueeze(1).expand(-1, len_b, -1, -1)
        K_batch = K.unsqueeze(0).expand(len_a,-1, -1, -1).transpose(-1,-2)
        scores = torch.matmul(Q_batch, K_batch)
        scores = scores * scale
        

        aidx_tensor = torch.tensor(aidx, device=scores.device).unsqueeze(1)  # (M, 1)：每行对应原始索引
        bidx_tensor = torch.tensor(range(K_batch.shape[1]), device=scores.device).unsqueeze(0)  # (1, N)：每列对应原始索引
        # 2.2 因果约束：bidx[n] ≤ aidx[m] → 掩码为True（保留），否则False（设为-inf）
        causal_mask = (bidx_tensor <= aidx_tensor)  # shape=(M, N)，bool型
        
        # 步骤3：扩展掩码到 (M, N, a, b) 维度（匹配切片张量）
        causal_mask = causal_mask.unsqueeze(-1).unsqueeze(-1)  # (M, N, 1, 1)
        causal_mask = causal_mask.expand(-1, -1, scores.shape[2], scores.shape[3])  # (M, N, a, b)
        
        # 步骤4：应用掩码：False位置设为 -inf
        scores = scores.masked_fill(~causal_mask, -float('inf'))
        
        
        # print(scores)
        if normalize:
            scores = F.softmax(scores, dim=1)
        attention_mean_k = scores[:,bidx,:,:].mean(-1)
        attention_mean=attention_mean_k.mean(-1).float().detach().cpu().numpy()
        
        return attention_mean
    def attention1(self,id1,id2,data,signal,save_path='/data/yulin/Bagel/result_tmp/',layers=[]):
        plt.close('all')
        save_path=os.path.join(save_path,f"{id1}_to_{id2}.png")
        if 'TEXT' not in id1 and 'IMAGE' not in id1:id1=[int(id1)]
        else:
            pic_indices = []
            for idx, sig in enumerate(signal):
                if isinstance(sig, str) and id1 in sig:
                    pic_indices.append(idx)
                elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                    pic_indices.append(idx)
            id1=pic_indices
        if 'TEXT' not in id2 and 'IMAGE' not in id2:id2=[int(id2)]
        else:
            pic_indices = []
            for idx, sig in enumerate(signal):
                if isinstance(sig, str) and id2 in sig:
                    pic_indices.append(idx)
                elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                    pic_indices.append(idx)
            id2=pic_indices
        if len(layers)==0:
            x=[t for t in range(int(len(data)/2))]
            layers=range(int(len(data)/2))
        else:
            x=layers
        y=[]
        
        for t in layers:
            print(f'calculating layer_id={t}/{layers}')
            qname=self.prefix+str(t)+self.qfix
            kname=self.prefix+str(t)+self.kfix
            q=data[qname]['output']
            k=data[kname]['output']
            y.append(self.compute_batch_attention(q,k,id1,id2).mean())
        print(x,y)
        plt.figure(figsize=(12, 6))
        
        # 绘制折线图
        plt.plot(x, y, 'b-o', linewidth=2, markersize=8, 
                markerfacecolor='red', markeredgecolor='red', 
                markeredgewidth=2)
        
        # 添加标题和标签
        plt.title(f'Attention from Token {id1} to Token {id2} Across Layers', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Layer Index', fontsize=14)
        plt.ylabel('Attention Value', fontsize=14)
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 在每个数据点上添加数值标签
        for xi, yi in zip(x, y):
            plt.annotate(f'{yi:.4f}', 
                        xy=(xi, yi), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=10,
                        color='darkblue')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            print(f"折线图已保存到: {save_path}")
        
        return
    def attention2(self,id,pic,pic_path,data,signal,save_path='/data/yulin/Bagel/result_tmp/',layers=[]):
        plt.close('all')
        save_path=os.path.join(save_path,f"{id}_to_{pic}.png")
        if 'TEXT' not in id and 'IMAGE' not in id:id=[int(id)]
        else:
            pic_indices = []
            for idx, sig in enumerate(signal):
                if isinstance(sig, str) and id in sig:
                    pic_indices.append(idx)
                elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                    pic_indices.append(idx)
            id=pic_indices
        pic_indices = []
        for idx, sig in enumerate(signal):
            if isinstance(sig, str) and pic in sig:
                pic_indices.append(idx)
            elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                pic_indices.append(idx)
        
        if not pic_indices:
            print(f"警告: 在signal中未找到图片 '{pic}' 对应的token索引")
            return None
        pic_indices=pic_indices[1:-1]
        print(f"找到 {len(pic_indices)} 个图像token索引: {pic_indices[:10]}...")
        
        # 2. 确定总层数
        total_layers = int(len(data) / 2)
        x = list(range(total_layers))
        
        # 3. 计算每一层中id到每个图像token的注意力
        # 存储格式: [层数, 图像token数]
        layer_attention=[]
        if len(layers)==0:
            layers=range(int(len(data)/2))
        for layer_id in layers:
            print(f'calculating layer_id={layer_id}/{layers}')
            qname = self.prefix + str(layer_id) + self.qfix
            kname = self.prefix + str(layer_id) + self.kfix
                
                
            q = data[qname]['output']  # [seq_len, hidden_dim]
            k = data[kname]['output']  # [seq_len, hidden_dim]
            layer_attention.append(self.compute_batch_attention(q,k,id,pic_indices).mean(axis=0))
        layer_attention_array=np.array(layer_attention)
        layer_attention=layer_attention_array.mean(axis=0)
        print(len(layer_attention))
        # 5. 加载原始图像
        try:
            # 尝试用PIL打开
            img = Image.open(pic_path)
            img_array = np.array(img)
            
            # 如果是RGBA，转换为RGB
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
        except Exception as e:
            print(f"用PIL打开图像失败: {e}，尝试用OpenCV")
            try:
                img_array = cv2.imread(pic_path)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                print(f"用OpenCV打开图像也失败: {e2}")
                return None
        
        # 6. 将图像token注意力映射到图像空间
        # 假设图像token是按patch顺序排列的
        h_patches = int(np.sqrt(len(pic_indices)))  # 假设是正方形patch网格
        if h_patches * h_patches != len(pic_indices):
            # 如果不是完全平方数，尝试推断
            h_patches = int(np.sqrt(len(pic_indices)))
            w_patches = int(np.ceil(len(pic_indices) / h_patches))
        else:
            w_patches = h_patches
        
        print(f"推断patch网格: {h_patches}×{w_patches}")
        
        # 创建注意力热力图（patch级别）
        attention_map_patches = np.zeros((h_patches, w_patches))
        
        for i, pic_idx in enumerate(pic_indices):
            if i >= h_patches * w_patches:
                break
            row = i // w_patches
            col = i % w_patches
            attention_map_patches[row, col] = layer_attention[i]
        
        # 7. 将patch级别的热力图扩展到像素级别
        img_height, img_width = img_array.shape[:2]
        patch_height = img_height // h_patches
        patch_width = img_width // w_patches
        
        attention_map_full = np.zeros((img_height, img_width))
        
        for i in range(h_patches):
            for j in range(w_patches):
                if i * w_patches + j < len(pic_indices):
                    start_h = i * patch_height
                    end_h = (i + 1) * patch_height if i < h_patches - 1 else img_height
                    start_w = j * patch_width
                    end_w = (j + 1) * patch_width if j < w_patches - 1 else img_width
                    
                    attention_map_full[start_h:end_h, start_w:end_w] = attention_map_patches[i, j]
        
        # 8. 创建可视化图像（仅保留原图+热力图叠加的子图）
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # 仅保留原图+热力图叠加的部分
        ax.imshow(img_array, alpha=0.7)
        heatmap = ax.imshow(attention_map_full, cmap='hot', alpha=0.5, 
                            vmin=layer_attention.min(), vmax=layer_attention.max())
        ax.set_title(f'Attention Heatmap (Layer {layer_id})', fontsize=14)
        ax.axis('off')
        plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        
        # 添加整体标题
        fig.suptitle(f'Token {id} → Image "{pic}" Attention Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # 9. 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"热力图已保存到: {save_path}")
        
        plt.close(fig)
        
        # 10. 返回统计信息
        stats = {
            'image_token_indices': pic_indices,
            'layer_attention': layer_attention,
            'mean_attention_per_layer': layer_attention_array.mean(axis=1),  # 补充原代码中缺失的变量定义
            'attention_map_shape': attention_map_full.shape,
            'max_attention': layer_attention.max(),
            'min_attention': layer_attention.min(),
            'mean_attention': layer_attention.mean(),
            'save_path': save_path
        }
        
        return stats
    def attention3(self,id1,id2,data,signal,save_path='/data/yulin/Bagel/result_tmp/',layers=[]):
        plt.close('all')
        save_path=os.path.join(save_path,f"Token:{id1}_to_{id2}.png")
        if 'TEXT' not in id1 and 'IMAGE' not in id1:id1=[int(id1)]
        else:
            pic_indices = []
            for idx, sig in enumerate(signal):
                if isinstance(sig, str) and id1 in sig:
                    pic_indices.append(idx)
                elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                    pic_indices.append(idx)
            id1=pic_indices
        if 'TEXT' not in id2 and 'IMAGE' not in id2:id2=[int(id2)]
        else:
            pic_indices = []
            for idx, sig in enumerate(signal):
                if isinstance(sig, str) and id2 in sig:
                    pic_indices.append(idx)
                elif isinstance(sig, dict) and 'content' in sig and pic in str(sig['content']):
                    pic_indices.append(idx)
            id2=pic_indices
        x=id2
        y=[]
        if len(layers)==0:
            layers=range(int(len(data)/2))
        for t in layers:
            print(f'calculating layer_id={t}/{layers}')
            qname=self.prefix+str(t)+self.qfix
            kname=self.prefix+str(t)+self.kfix
            q=data[qname]['output']
            k=data[kname]['output']
            y.append(self.compute_batch_attention(q,k,id1,id2).mean(axis=0))
        y_array=np.array(y)
        y=y_array.mean(axis=0)
        plt.figure(figsize=(10, 5))
        
        # 绘制折线图
        plt.plot(x, y, 'b-o', linewidth=2, markersize=8, 
                markerfacecolor='red', markeredgecolor='red', 
                markeredgewidth=2)
        
        plt.xlabel('Token Index', fontsize=14)
        plt.ylabel('Attention Avg. Value', fontsize=14)
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 在每个数据点上添加数值标签
        for xi, yi in zip(x, y):
            plt.annotate(f'{yi:.4f}', 
                        xy=(xi, yi), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=10,
                        color='darkblue')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            print(f"折线图已保存到: {save_path}")
        
        return plt.gcf()

# Q=torch.randn(5,512)
# K=torch.randn(5,256)
# Scorer=MOTAttentionCapture()
# print(Q)
# print(K)
# print(Scorer.compute_batch_attention(Q,K,[0,2,4],[1,3]).mean())
# plt.figure(figsize=(12, 6))
