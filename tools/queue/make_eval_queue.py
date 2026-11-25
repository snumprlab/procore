"""Generate eval jobs (with dependency files) for the LMM-ICL table + cross-dataset evals."""
import sys
V = '/tmp/procore/checkpoints_verified'
DS = ['truemicl', 'mic', 'vliclv2', 'isekai', 'pororov4', 'storystreamv2', 'flintstonesv3', 'coco']
CROSS = [('truemicl', 'isekai'), ('isekai', 'truemicl'), ('mic', 'vliclv2'), ('vliclv2', 'mic')] + \
        [('coco', p) for p in ['pororov4', 'storystreamv2', 'flintstonesv3']] + \
        [(p, 'coco') for p in ['pororov4', 'storystreamv2', 'flintstonesv3']]
DARA_K = {'isekai': [2, 4, 8, 12], 'truemicl': [1, 2, 4, 8], 'mic': [1, 2, 4, 8], 'vliclv2': [1, 2, 4, 8], 'coco': [1, 2, 4, 8],
          'pororov4': [1, 2, 3, 4], 'storystreamv2': [1, 2, 3, 4], 'flintstonesv3': [1, 2, 3, 4]}
def ck(note): return f'{V}/client_states_{note}/server_model_round0.pth'
def icl_needs(m, a, src):
    if m == 'mt-icl': return [ck(f'sft_{a}_{src}_1_mixed_scen0')]
    if m == 'mimic_mixed': return [ck(f'mimic_{a}_{src}_1_mixed_scen0_5e-3')]
    if m == 'stv': return [ck(f'stv_{a}_{src}_1_mixed_scen0')]
    if m == 'dara': return [ck(f'dara_{a}_{src}_1_original_scen{k}_1e-3') for k in DARA_K[src]]
def tv_need(m, a, src):
    return [f'/tmp/procore/MTV/storage_v2/{a}/{src}/mtv_pooled.pt' if m == 'mtv_v2' else f'/tmp/procore/fv_v2/{a}/{src}/fv_pooled.pt']
out = {}
for a in ['qwen3-8b', 'onevision1_5-8b', 'internvl3_5-8b', 'idefics-8b']:
    J = []
    for d in DS:
        base = f'MODEL_ARCHS={a} DATASET_NAME={d}'
        J.append(f'{base} METHOD=None ORDERS="random mmices_v2 rices_v2" :: eval')
        J.append(f'{base} VARIANT=vicl ORDERS=vicl_v2 :: eval')
        for m in ['ibd', 'sparc']: J.append(f'{base} METHOD={m} :: eval')
        for m, n in [('Finetune', 'inpaintcoco'), ('lrv', 'lrv'), ('volcano', 'volcano'), ('lure', 'lure')]:
            J.append(f'{base} METHOD={m} :: eval :: {ck(f"sft_{a}_{n}_1_original_scen2")}')
        for m in ['mtv_v2', 'fv_v2']: J.append(f'{base} METHOD={m} :: eval :: {" ".join(tv_need(m, a, d))}')
        for m in ['mt-icl', 'mimic_mixed', 'stv', 'dara']: J.append(f'{base} METHOD={m} :: eval :: {" ".join(icl_needs(m, a, d))}')
    for src, tgt in CROSS:
        base = f'MODEL_ARCHS={a} DATASET_NAME={tgt}'
        for m in ['mt-icl', 'mimic_mixed', 'stv', 'dara']: J.append(f'{base} TRAIN_DS={src} METHOD={m} :: eval :: {" ".join(icl_needs(m, a, src))}')
        for m in ['mtv_v2', 'fv_v2']: J.append(f'{base} TASKVEC_SRC_BENCH={src} METHOD={m} :: eval :: {" ".join(tv_need(m, a, src))}')
    out[a] = J
G = {'qwen3-8b': ['RW1_2_4', 'RW2_1_1']}
for rw in G['qwen3-8b']:  # ProCoRe (inpaintcoco) on coco; the 7 other sets were evaluated by run_icl_eval.sh
    note = f'qwen3-8b_grpo_inpaintcoco_original_procore_propgpt_{rw}_numgen4_scen3_2e-5'
    out['qwen3-8b'].append(f'MODEL_ARCHS=qwen3-8b DATASET_NAME=coco METHOD=grpo GRPO_NOTE={note} :: eval :: {ck(note)}')
for a, J in out.items():
    open(f'/tmp/procore/tools/queue/eval_{a}.txt', 'w').write('\n'.join(J) + '\n'); print(a, len(J))
