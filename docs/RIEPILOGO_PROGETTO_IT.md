# Riepilogo completo del progetto (in italiano)

**Progetto:** un decoder DETR-like / in stile D4RT per **segmentazione 3D di istanze
consistente tra viste multiple (multi-view consistent)**, addestrato sopra il backbone
**VGGT-1B congelato**. La supervisione (ground truth) viene da maschere **SAM3** calcolate su
scene **ScanNet**. Il backbone VGGT non viene mai modificato: si addestrano solo i ~6,5M di
parametri della testa.

Questo documento racconta *tutto* ciò che è stato fatto finora: l'architettura, le
**motivazioni** dietro ogni scelta, gli esperimenti con i loro risultati e — soprattutto —
come vanno interpretati. È aggiornato al **7 luglio 2026**. Le fonti sono
`docs/MILESTONES.md` (riassunto consolidato), `docs/todo.md` (lavoro aperto) e
`docs/old/` (narrativa completa per milestone).

---

## 1. Obiettivo e idea di fondo

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) è un modello feed-forward di
ricostruzione 3D: prende S immagini di una scena e, in un solo passaggio, produce feature che
codificano la geometria 3D della scena fusa tra tutte le viste. Il progetto **non modifica
VGGT**: lo usa come estrattore di feature multi-vista congelato e ci attacca sopra una testa
leggera e addestrabile che risolve un compito nuovo — **segmentazione di istanze coerente tra
le viste**: lo stesso oggetto fisico deve ricevere la stessa identità (stessa maschera, stesso
ID) in tutti i frame in cui appare.

Perché questa impostazione:

- **Backbone congelato** → si addestrano solo ~6,5M di parametri contro ~1,26 miliardi del
  backbone. Il training diventa questione di *minuti* (le feature del backbone si calcolano una
  volta sola e si mettono in cache), gli esperimenti diventano economici e iterabili, e ogni
  miglioramento è attribuibile alla testa, non a un fine-tuning del backbone.
- **Consistenza multi-vista "per costruzione"**: nel design scelto, una query del decoder
  emette **una sola maschera che copre congiuntamente tutti gli S frame**
  (`pred_masks [B, N, S, h, w]`). Non esiste un passo di detection per-frame seguito da un
  linking di ID che potrebbe rompersi: stesso colore = stessa query = stessa istanza, per
  costruzione. Questo è il punto di forza architetturale del progetto.
- **Supervisione da SAM3** (pseudo-label, non annotazioni manuali): scala facilmente a
  centinaia di scene; la consistenza cross-frame degli ID di istanza viene dal video tracking
  di SAM3.

---

## 2. Il backbone VGGT e il punto di aggancio (hook)

### Come funziona VGGT (in breve)

`vggt/models/vggt.py::VGGT` incapsula `vggt/models/aggregator.py::Aggregator`: 24 blocchi di
**attenzione alternata** — self-attention per-frame (token `[B·S, P, C]`) alternata ad
attenzione globale cross-frame (`[B, S·P, C]`). A certi layer (indici 4, 11, 17, 23)
l'aggregatore concatena le feature per-frame e quelle globali lungo il canale
(`torch.cat([frame, global], dim=-1)`), producendo token di dimensione `2C = 2048`. Sopra
l'aggregatore stanno le teste originali (camera, depth, point, track) — irrilevanti per
questo progetto, come la cartella `training/` (framework di finetuning upstream su Co3D).

### Dove si aggancia la testa e perché

**Hook scelto:** `aggregated_tokens_list[-1]` (l'ultimo layer messo in cache, indice 23) →
le **feature globali di scena** `F : [B, S, P, 2048]`, dove `P` = token patch + 1 token camera
+ 4 register token (`patch_start_idx` li separa; con immagini 518×518 e patch 14×14 si hanno
37×37 = 1369 patch token).

Motivazioni (da `docs/HOOK_PLAN.md`):
- Ogni token `F[b, s, p, :]` contiene sia feature locali del frame (1024 dim) sia feature
  globali cross-vista (1024 dim): è esattamente la "memoria" ricca di informazione 3D che un
  decoder a cross-attention vuole.
- Le feature sono già calcolate e cache-abili: **zero modifiche all'Aggregator** (la parte
  computazionalmente costosa).
- La testa è un modulo separato: si può aggiungere/rimuovere senza toccare VGGT, e il backbone
  può restare congelato (`no_grad`).
- I patch token mappano a posizioni spaziali dell'immagine → si possono recuperare le
  coordinate (u, v), essenziale sia per le query a punto sia per la testa densa di maschere.

---

## 3. Dataset e ground truth

### Struttura

- Le scene ScanNet stanno in
  `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/<scene>/raw_data`.
  Oggi: **200 scene** (scene0000–0199), **≈4195 istanze**.
- Ogni scena ha `color/` (>5500 frame grezzi, **non usati**), `subset/` (i ~100 frame a
  stride 5 che hanno effettivamente le maschere — il loader carica da qui), `masks/<classe>/`
  (maschere binarie per-classe) e `masks_instance/<classe>_<k>/` (maschere binarie
  **per-istanza**, una cartella per oggetto, `k` = indice per-classe a base zero, es.
  `chair_0`, `chair_1`).
- 19 classi ScanNet mappate agli indici `1..19`; l'indice `0` è ovunque il background.
- Le immagini sono ridimensionate a **518×518** perché l'input di VGGT deve essere divisibile
  per la patch size 14; maschere e valutazione vivono alla risoluzione della griglia di patch
  **37×37** (con `--mask_upsample` si può salire, vedi §9.6).

### Dalla GT per-classe alla GT per-istanza (una decisione chiave)

All'inizio esistevano solo maschere **binarie per-classe**: impossibile separare due sedie
diverse. Il loader assegnava quindi **un solo ID globale per classe**, consistente tra le
viste — il massimo che quei label permettevano (limite dei *dati*, non del codice; §8.3 di
MILESTONE_1). Conseguenza imbarazzante fatta notare dal supervisore: con al massimo
un'istanza per classe, colorare "per istanza" e "per categoria" era visivamente
indistinguibile — nessuna figura poteva *dimostrare* la separazione di istanze della stessa
classe.

La decisione (meeting 12 giugno): rifare la run SAM3 emettendo **una maschera binaria per
istanza**, con identità cross-frame dal video tracking di SAM3. Le classi "stuff"
(`wall`, `floor`) restano una singola istanza `_0`; l'unione delle istanze di una classe
riproduce la vecchia maschera per-classe (union-IoU ≈ 1.0 — un check di sanità del
preprocessing). Il loader ha guadagnato il flag `instance_level` (CLI `--instance_level`):
ogni cartella `masks_instance/<classe>_<k>` diventa un'istanza globale distinta e `classes`
ripete gli indici di classe. **Matcher, loss e metriche non hanno richiesto alcuna modifica**
— erano già "Hungarian sulle istanze", una conferma della bontà del design modulare.

### Logistica dei dati (importante sul cluster)

Leggere migliaia di piccoli PNG dal filesystem di rete (`work`) è lentissimo. Tutto il
dataset viaggia quindi come **un singolo tar zstd**
`scannet_instance_dataset_full.tar.zst` (~2,6 GB compresso, ~5,4 GB scompattato): ogni job
SLURM lo copia sullo scratch locale del nodo (`$TMPDIR`), lo scompatta una volta e legge
dall'SSD locale (`slurm/stage_dataset.sh` esporta `SCANNET_ROOT=$TMPDIR/scans`;
`train_multiscene.py` lo usa come `--scans_root` di default; header SLURM con
`--tmp=16000` MB).

---

## 4. Architettura della testa di segmentazione

Pipeline (un componente per file, ciascuno con il suo test standalone eseguibile su CPU):

```
Immagini [B, S, 3, 518, 518]
        ▼
VGGT Aggregator (congelato)  →  F : [B, S, P, 2048]        ← "memory" della cross-attention
        ▼
QueryGenerator  →  Query [B, N, 256]                        ← "tgt" della cross-attention
        ▼
InstanceDecoder (nn.TransformerDecoder, 4 layer × 8 teste; memoria normata + skip delle query)
        ├─► class_head      → logit di classe  [B, N, 20]   (19 classi + background)
        └─► mask_embed_head → embedding di maschera [B, N, 256]  (kernel per-query)
                  │  similarità coseno · feature per-pixel (dai patch token VGGT)
                  ▼
              pred_masks [B, N, S, h, w]   (logit densi di maschera per ogni frame)
        ▼
PointBipartiteMatcher (Hungarian, costo Dice+BCE mask-aware) → coppie (pred, gt)
        ▼
D4RTLoss = Focal (classe) + Dice + BCE pesata sul foreground  [+ no-object loss opzionale]
        ▼
Metriche: mIoU / AP50 / AP75 / mAP / class_acc
```

### 4.1 `data/scannet_overfit.py` — il loader

`ScanNetSingleSceneDataset` / `ScanNetMultiSceneDataset`. Restituisce: `images`, `masks`
(mappa di ID di istanza per pixel, **consistente tra i frame**), `classes` (classe di ogni
istanza globale), `coordinates` (centroide (u,v) normalizzato dell'istanza nel suo frame
"rappresentativo" = quello con area massima), `frame_ids`, `instance_ids`.

### 4.2 `models/d4rt_decoder.py` — query e decoder

**`QueryGenerator`** — tre modalità (`query_mode`, salvato nel `head_config` del checkpoint):
- **`point`** (default storico): ogni query è un *prompt a punto* = somma di
  (a) encoding posizionale di Fourier di (u,v) (16 frequenze log-spaced → 64 dim),
  (b) embedding di vista appreso (`nn.Embedding(num_views, 256)`),
  (c) MLP su una patch RGB 9×9 estratta con `grid_sample` intorno a (u,v) nella vista giusta.
  I tre contributi vengono proiettati a 256, sommati e riproiettati.
- **`learned`**: vere object query DETR (`nn.Embedding(M, 256)`), nessuna coordinata; il
  matcher usa `coord_weight=0`. **Oggi è la modalità base del progetto** (vedi §9.4).
- **`hybrid`**: i primi M slot appresi, il resto query a punto.

**`InstanceDecoder`**: proietta la memoria `F` da 2048 → 256, la **normalizza con LayerNorm**,
poi un `nn.TransformerDecoder` standard (4 layer, 8 teste) con le query come `tgt` e la
memoria come `memory`, seguito da una **skip connection** `decoded = decoded + queries`.
Teste in uscita: `class_head` (20 logit), `mask_embed_head` (kernel di maschera per query) e la
testa densa in stile Mask2Former: un `mask_feature_proj` trasforma i patch token VGGT in una
mappa di feature per-pixel, e la maschera di ogni query è la **similarità coseno** del suo
embedding con quella mappa (temperatura apprendibile) → `pred_masks [B, N, S, 37, 37]`.

Con `mask_upsample 2/4` la mappa passa prima per `models/mask_upsampler.py::MaskUpsampler`
(stadi bilinear + conv 3×3 + GroupNorm + ReLU) → 74×74 o 148×148, con la GT costruita alla
stessa risoluzione.

### 4.3 `train/loss.py` — matching e loss

- **`PointBipartiteMatcher`**: matrice di costo = costo di classe (1 − p_corretta) + L2 sulle
  coordinate (peso 0 con query apprese) + **costo di maschera** Dice+BCE denso (stile
  Mask2Former); poi `scipy.optimize.linear_sum_assignment` (Hungarian). Il costo è protetto
  con `nan_to_num` (fix del 2026-07-03, vedi arm D).
- **`D4RTLoss`**: Focal loss (α=0.25, γ=2.0) sulla classe + Dice + BCE **pesata sul
  foreground** (`pos_weight`, altrimenti le maschere sparse collassano a vuote) sulle maschere
  matchate. Opzionale (default 0.1 nello script): **no-object loss** DETR — la loss di classe
  gira su *tutte* le N query, le non matchate vengono spinte verso il background con peso
  ridotto. Batch-aware: con B>1 la GT è una lista di tensori per-sample e l'Hungarian gira per
  sample (stile DETR).
- Le **coordinate sono prompt, non predizioni**: entrano nel costo del matcher ma non hanno
  alcun termine di loss (decisione esplicita, §8.5 di MILESTONE_1).

### 4.4 `train/eval_metrics.py` — metriche

mIoU / AP50 / AP75 / mAP / class_acc, riportate in **due regimi**:
- **prompted**: query ai centroidi GT — "dato un punto sull'oggetto, segmentalo e
  classificalo";
- **unprompted**: griglia uniforme di query (default 6×6 per frame), zero informazione GT —
  "trova tu le istanze". **L'AP50 unprompted su validation (val[grid] AP50) è il numero
  di detection onesto del progetto** (vedi §6).

### 4.5 `scripts/train_multiscene.py` — il loop di training

Il trucco di efficienza che rende tutto il progetto iterabile: il backbone congelato gira
**una sola volta per "bundle" di scena all'inizio** e le sue feature vengono messe in cache;
ogni epoca esegue solo la testa da 6,5M parametri. Risultato: 1000 epoche su 50 scene ≈
minuti, non ore. `--cache_device cpu` sposta la cache in RAM host e rimuove il limite di
memoria GPU sul numero di scene.

---

## 5. I vincoli "conquistati sul campo" (hard-won constraints)

Questa è la parte più preziosa della narrativa di debugging (MILESTONE_1 §6). Il primo
tentativo di testa densa **non imparava nulla**: la loss scendeva ma il mIoU restava 0.
L'instrumentazione passo-passo mostrò il **collasso dell'output del decoder**: query distinte
(std ≈ 2) producevano vettori decodificati *identici* (std ≈ 0) — tutte le istanze
condividevano una sola maschera. Causa: le feature grezze di VGGT hanno magnitudini enormi e
la memoria non normalizzata domina il residual stream della cross-attention, schiacciando
l'identità delle query. Un dettaglio subdolo: una precedente loss proxy (L2 sugli embedding)
aveva *mascherato* il problema fornendo target distinti per query; è stata la vera mask loss a
esporlo.

Da qui i vincoli, che **non vanno mai violati** (romperli fa fallire il training in silenzio):

1. **LayerNorm sulla memoria proiettata** + **skip connection delle query** nel decoder —
   altrimenti tutte le query collassano sullo stesso vettore.
2. I logit di maschera usano la **similarità coseno con temperatura apprendibile**, non
   prodotti scalari grezzi (le norme enormi saturerebbero le sigmoidi).
3. La BCE usa un **`pos_weight` sul foreground**; il **gradient clipping** è attivo.
4. Le **coordinate sono prompt**, mai un termine di loss.
5. Un test di overfit deve tenere fissi **input e target** per tutte le epoche, altrimenti non
   misura nulla.

---

## 6. Il protocollo di valutazione e le sue lezioni

Alcuni principi metodologici emersi strada facendo, oggi codificati nel progetto:

- **AP50 unprompted (val[grid]) è il numero onesto.** Il mIoU unprompted è *ottimista*: con
  ~100+ candidati dalla griglia, le coppie matchate beneficiano della scelta ampia mentre i
  falsi positivi non matchati non vengono puniti dal mIoU (solo l'AP li punisce). Ovunque si
  riportino numeri unprompted va detto.
- **La metrica di selezione del checkpoint conta.** I checkpoint migliori per mIoU e per AP50
  cadono a epoche diverse in *tutti* i run → si salvano due best: `checkpoint_best.pth`
  (val mIoU prompted) e `checkpoint_best_ap50.pth` (val[grid] AP50).
- **Protocollo identico o confronto nullo.** I primi due run di scaling (giugno) produssero
  una curva *al contrario* (N=25 peggio di N=10): un artefatto. La run N=25 era stata fermata
  dall'early stopping all'epoca 200, ancora a LR di picco (la cosine copriva 1000 epoche),
  con pazienza dimezzata (contata in eval, non epoche, e `eval_interval` diverso) e su una
  val di sole 3 scene (rumore ±0.05–0.08 per eval). Lezioni operative
  (`docs/old/SCALING_RUNS_ANALYSIS.md`): early stopping **spento** per le curve (i run durano
  minuti, non c'è pressione di calcolo), `--eval_interval 50` ovunque, val **allargata a 10
  scene (scene0080–0089)**, `--schedule_epochs` per scollegare la lunghezza della cosine dal
  numero di epoche, e metriche persistite in `metrics.jsonl` (una riga JSON per eval) così i
  grafici leggono file e non log.
- **Val scenes 0080–0089 sono escluse da ogni train set.**

---

## 7. Milestone 1 — Il prototipo (COMPLETATA)

**Obiettivo:** costruire l'intera pipeline end-to-end e dimostrare che i gradienti fluiscono
correttamente dalle pseudo-label SAM3, attraverso il matching, fino al decoder — con il
backbone congelato. Sviluppo rigorosamente a fasi (1–6), ognuna validata da un test
standalone prima di procedere (regola di lavoro del progetto fin dal brief originale).

**Risultati chiave:**

- **Overfit su singola scena** (scene0000_00, 4 frame, 16 query, 11 istanze cross-view, 400
  epoche): loss −88,6%, tutte le 11 istanze matchate ogni epoca, e soprattutto metriche vere:
  **mIoU 0.004 → 0.900, AP50 0 → 0.962, class_acc → 1.0**. Non "la loss scende" ma un overfit
  misurabile — dopo aver risolto il collasso del decoder (§5).
- **Training multi-scena** (4 scene, 2000 epoche): **mIoU medio di train 0.967**, class_acc
  0.94 — un solo set di pesi rappresenta più scene simultaneamente.
- **Ma zero generalizzazione**: sulla 5ª scena mai vista, mIoU finale **0.027**, con un picco
  ~0.13 a metà training poi decaduto (overfitting classico, visibile proprio grazie alla
  scena held-out).

**Interpretazione e conseguenze (il "perché" di Milestone 2):** il picco a metà training
dimostrava che *il segnale di generalizzazione esiste* ma andava catturato (→ best checkpoint
+ early stopping); l'AP50 di train basso (~0.54) nonostante mIoU 0.97 rivelava che le query di
background non venivano mai spinte al background (→ no-object loss DETR); il batch fisso per
scena era un overfit deliberato (→ augmentation); e 4 scene sono troppo poche (→ scaling).

---

## 8. Milestone 2 — Training regolarizzato e non-promptato (COMPLETATA)

**Obiettivo:** trasformare la pipeline di overfit in un vero loop di training la cui testa sia
usabile **senza prompt GT** a inference.

**Cosa è stato aggiunto e perché:**

1. **No-object loss** (stile DETR, `no_object_weight`, default 0.1). Problema: solo le query
   matchate ricevevano una loss di classe, quindi il modello non imparava mai a dire "qui non
   c'è niente" — l'AP era schiacciato da detection spurie e l'inference *richiedeva* query
   ordinate secondo la GT. Soluzione: loss di classe su tutte le N query; le non matchate →
   background, sotto-pesate (0.1) per non annegare le poche matchate. Effetto: AP50 prompted
   di train **0.54 → 0.77**.
2. **Inference/eval unprompted su griglia** (`generate_grid_queries`, 6×6 per frame): reso
   possibile proprio dalla no-object loss. Da qui in poi ogni eval riporta le due colonne
   prompted/unprompted.
3. **Regolarizzazione compatibile con la cache**: `--bundles_per_scene K` (bundle 0
   deterministico per eval/checkpoint, gli altri con frame casuali — ogni bundle paga un solo
   passaggio di backbone), `--query_jitter` (i centroidi vengono perturbati ogni step: il
   modello non può memorizzare le posizioni esatte), ricampionamento del background a ogni
   step, `--color_jitter` (applicato *prima* del passaggio di backbone, così le feature in
   cache restano coerenti con le immagini salvate), `--cache_device cpu`.
4. **Selezione del modello**: `checkpoint_best.pth` sul val mIoU + early stopping opzionale —
   la risposta diretta al picco perso di M1.
5. **Retro-compatibilità totale**: il comportamento M1 si recupera esattamente con
   `--no_object_weight 0 --bundles_per_scene 1 --query_jitter 0 --fixed_bg` (convenzione di
   progetto: ogni nuova opzione ha default = comportamento precedente, così i test esistenti
   passano invariati).

**Risultato di validazione (5 scene, run `d4rt_m2_5scenes_20260610_133100`):** best val mIoU
**0.138** @ep450 (il valore finale era decaduto a 0.109 — il pattern di overfitting M1, ora
*catturato* dal best checkpoint invece che perso). Con zero informazione GT le query a griglia
raggiungevano su train mIoU 0.678, alla pari del prompted (0.666): l'inference unprompted era
diventata reale. L'AP50 unprompted restava basso perché più celle della griglia sparavano
sullo stesso oggetto (duplicati) — problema annotato che diventerà centrale in M3.

**Conclusione:** codice pronto a scalare, ma **gli esperimenti di scaling erano bloccati dai
dati** (servivano decine-centinaia di scene preprocessate con SAM3).

---

## 9. Milestone 3 — Scaling, modalità di query, pixel decoder, GT per-istanza

Milestone lunga e ricca, guidata da due input: l'analisi critica dei primi run di scaling
(§6) e il **feedback del supervisore del 12 giugno**, i cui cinque punti hanno di fatto
dettato l'agenda:

1. *"Stesso colore ≠ stessa istanza tra i frame"* → in realtà l'identità era architetturale
   (una query = una maschera multi-frame), ma con GT per-classe era **indimostrabile** → GT
   per-istanza (§3).
2. *"Risolvete le maschere 37×37 alla MaskDINO"* → pixel decoder (§9.6).
3. *"Con l'Hungarian matching in training non serve NMS — è DETR"* → giusto, ma il nostro
   training non esercitava mai quel meccanismo: le query a griglia esistevano **solo a eval**,
   quindi due query non stavano quasi mai sullo stesso oggetto in training e la no-object loss
   non vedeva mai una query "sull'oggetto ma ridondante". Da qui `--train_grid_queries`
   (arm B).
4. *"Formato label della prossima run SAM3: instance"* → deciso e fatto.
5. *"Point prompt vs learned object query: fate l'ablation"* → arms A/B/C/D.

### 9.1 Fase 0 — Strumentazione (CPU, FATTA 2026-06-14)

`metrics.jsonl` per run; checkpoint più piccoli (immagini **uint8**, 4× più piccole;
`--checkpoint_light` che non salva i pixel — a N=50 i checkpoint float superavano 1,6 GB e
venivano riscritti a ogni nuovo best); early stopping robusto al rumore (media mobile +
min-delta + rifiuto prima di metà schedule; spento nei run delle curve);
`checkpoint_best_ap50.pth`; `--schedule_epochs`; rifinitura visualizzazioni (legenda
`"{classe} #{k}"`, `--score_threshold` esposto); protocollo SLURM identico su tutti gli N.

### 9.2 La curva di scaling con prompt a punto (arm A) — e il suo plateau

Con GT per-istanza e la val larga (scene0080–0089), protocollo identico, 1000 epoche piene:

| N   | val mIoU  | val[grid] AP50 (onesto) | train mIoU finale |
|-----|-----------|-------------------------|-------------------|
| 10  | 0.152     | 0.089                   | 0.526             |
| 25  | 0.174     | 0.111                   | 0.353             |
| 50  | 0.212     | **0.125**               | 0.338             |
| 100 | **0.228** | 0.103                   | 0.272             |
| 200 | 0.216     | 0.105                   | 0.265             |

**Interpretazione:**
- Fino a N=50 entrambe le colonne salivano monotone (l'inversione a N=25 vista con 3 scene di
  val era rumore, sparita allargando la val — conferma che il fix del protocollo era dovuto).
- Da N=100 in poi la curva è **piatta**: val mIoU ~0.21–0.23, AP50 onesto ~0.10 (sotto il picco
  0.125 di N=50). **Più scene non sono più la leva.**
- Il gap train−val si restringe con N (0.37 → 0.05): il modello **non sta più overfittando** —
  ha colpito un tetto di capacità/architettura, non di dati. È questo che motiva lo
  spostamento sulle leve architetturali (query apprese, pixel decoder) invece che su altri
  dati.
- I numeri su GT per-istanza sono ≈ metà degli equivalenti per-classe: **il costo previsto del
  task più difficile** (più oggetti, più piccoli — le sedie si spezzano in tanti oggetti), non
  una regressione. Era stato dichiarato in anticipo nelle slide proprio per non farlo leggere
  come un peggioramento.

### 9.3 L'ablation delle modalità di query (arms A/B/C/D, N=50, GT per-istanza)

| Arm | Query di training | val mIoU | val[grid] AP50 | train mIoU | Esito |
|-----|-------------------|----------|----------------|------------|-------|
| A punto | centroidi GT + bg | 0.212 | 0.125 | 0.338 | baseline |
| B griglia | + griglia con offset casuale | **0.047** | 0.146 | **0.055** | apprendimento maschere collassato |
| C learned | M embedding appresi | **0.259** | **0.146** | 0.749 | miglior val; overfitta (gap 0.49) |
| D hybrid | learned + centroidi | ~0.27\* | — | 0.54\* | crash (NaN ~ep555) |

\* ultima eval di D prima del crash.

**Interpretazioni (le più importanti del progetto):**
- **C (query apprese) fu la sorpresa vincente** — *contro* il prior "le query DETR sono
  affamate di dati e sottoperformeranno con ≤50 scene". Il grosso caveat era il gap
  train−val di 0.49 (overfitting pesante): la domanda decisiva era il comportamento a N
  grandi.
- **B fallì in modo istruttivo**: la loss di classe scendeva ma il train mIoU restava ~0.05 —
  il modello imparava a classificare ma non a produrre maschere. Meccanismo diagnosticato: con
  ~320 query/step l'Hungarian instrada molte GT verso query di griglia, sotto-supervisionando
  le query a centroide usate dall'eval prompted, e la no-object loss su ~10× più query (quasi
  tutte background) sommerge i pochi gradienti delle maschere matchate. L'AP50 però *saliva*
  (l'effetto di soppressione duplicati voluto) → il fix ipotizzato: normalizzare il termine
  no-object per numero di query.
- **D era il braccio più promettente prima di morire**: NaN/inf nella matrice di costo di
  `linear_sum_assignment` ~ep555 (gradienti esplosivi nel percorso misto learned+point).

### 9.4 Arm C scalato a N=200 — il risultato headline (2026-06-22)

| Arm C learned | val mIoU | val[grid] AP50 onesto | train mIoU | gap |
|---------------|----------|------------------------|------------|-----|
| N=50 | 0.259 | 0.146 | 0.749 | 0.49 |
| N=200 best (@ep600) | **0.371** | 0.228 | 0.457 | **0.086** |
| N=200 finale (@ep1000) | 0.326 | **0.228** | 0.560 | 0.23 |

Run: `d4rt_full_inst_learned_20260622_183203`.

**Lettura:**
- **Le query apprese rompono il plateau: il tetto era la testa, non i dati.** Contro il
  baseline a punto N=200 (0.216 / 0.105): +0.15 di val mIoU e **AP50 onesto più che
  raddoppiato** (0.105 → 0.228).
- Le query apprese **continuano a scalare** dove i point prompt saturavano (0.259 → 0.371 da
  N=50 a N=200) e il loro overfitting a N=50 **si è risolto da solo con più dati** (gap
  0.49 → 0.086) — esattamente il crossover previsto ("data-hungry" significa che con più dati
  migliorano).
- Con le query apprese non esistono prompt, quindi prompted == grid: **0.228 è il numero di
  detection onesto e incondizionato del progetto.**
- **Da qui arm C è la base di default per ogni esperimento successivo**
  (`--query_mode learned --num_learned_queries 64 --instance_level`).

### 9.5 I fix di B e D e i rerun (fix 2026-07-03, esiti 2026-07-07)

Entrambi i bug sono stati corretti; nessuno dei due bracci batte arm C — chiusi entrambi.

| Rerun (N=50) | best val[grid] AP50 | best val[grid] mIoU | Esito |
|---|---|---|---|
| B `_gridq_fix` (`--no_object_norm matched`) | **0.161** @ep700 | 0.284 | collasso risolto (train[grid] mIoU 0.055 → 0.458); supera la soglia di successo ≥0.125 |
| D `_hybrid_fix` (`--learned_query_lr_scale 0.1`, grad-clip 0.5, matcher protetto) | 0.146 @ep200 | 0.247 @ep250 | NaN sparito (1000 epoche pulite), ma *pareggia* soltanto arm C a N=50 |

- Il fix di B (`--no_object_norm matched`) normalizza la no-object loss per termine
  (`matched.mean() + w·unmatched.mean()`) così le query di griglia appese non diluiscono più i
  gradienti delle matchate — esattamente il meccanismo diagnosticato in §9.3. B va giudicato
  solo sulle colonne grid: il suo mIoU prompted resta basso per instradamento delle GT verso
  le query di griglia, non per il vecchio collasso.
- **B scalato a N=190**: val[grid] mIoU **0.372** @ep1000 (pareggia arm C) ma AP50 onesto max
  **0.185** e instabile tra le eval (0.071 @ep1000) — ben sotto lo 0.228 di C. Le query di
  griglia allenate recuperano la qualità delle maschere a scala, ma non una detection stabile.
- **D non NaN-a più ma non vince**: i prompt a centroide reintroducono l'overfitting del
  percorso a punto che le query pure apprese avevano risolto (val decade 0.247 → 0.177 mentre
  il train[grid] sale a 0.75). Per la regola di decisione del progetto ("si scala solo su una
  vittoria"): niente run a N=190.
- **Verdetto: arm C (query apprese pure) resta la base.**

### 9.6 Fase 5 — Pixel decoder in stile MaskDINO (allenato 2026-06-30 — risultato neutro)

Idea (dal punto 2 del supervisore): la testa di maschere è già il *meccanismo*
Mask2Former/MaskDINO (embedding di query ⊗ mappa di feature via coseno); il pezzo mancante è
un pixel decoder che upsampli la mappa 37×37 prima del prodotto. `--mask_upsample 2` (74×74)
sulla base arm C a N=190:

| vs baseline us=1 | val[grid] AP50 onesto | val mIoU |
|------------------|------------------------|----------|
| us=1 | 0.228 | **0.371** |
| us=2 best | **0.236** @ep500 | 0.355 @ep250 |
| us=2 finale @ep1000 | 0.200 | 0.311 |

**Un pareggio**: +0.008 AP50, −0.016 mIoU — dentro il rumore tra run. **La risoluzione delle
maschere NON è il collo di bottiglia attuale**; `--mask_upsample 4` deprioritizzato (regola:
si insiste solo sulle vittorie). Implicazione importante: la confusione
window/door/picture è **semantica, non limitata dalla risoluzione** — le prossime leve sono
lo sweep del no-object weight / la soppressione dei duplicati, non maschere più nitide.

### 9.7 Sweep della soglia di score (2026-07-03 — negativo, si tiene 0.5)

La scoperta storica dell'"under-confidence" (molte predizioni corrette a score 0.28–0.49,
scartate dalla soglia 0.5) era un fenomeno **dei soli point prompt**: sul checkpoint arm C
N=200, abbassare la soglia a 0.3 aggiunge 76 istanze su val di cui solo 2 corrette — puro
rumore — e il modello già tiene 338 istanze contro 144 GT a soglia 0.5. Con le query apprese
il problema è l'**over**-prediction (duplicati/falsi positivi) → la leva è lo sweep del
no-object weight, non la soglia. Soglia confermata a 0.5.

---

## 10. Risultati qualitativi persistenti (da portarsi dietro)

- **Cluster di confusione di classe:** `window ↔ door ↔ picture ↔ curtain` — oggetti piatti,
  a muro, rettangolari; il pixel decoder non l'ha risolto → è confusione semantica.
- **Buchi di copertura:** i sanitari (`toilet`/`sink` → `chair`) quando le scene di train non
  ne contengono — si risolve con scene più varie (verificato migliorare con N).
- **Separazione di istanze della stessa classe dimostrata** visivamente (più sedie in colori
  diversi sulle scene di val) — la figura che il supervisore aveva chiesto.
- Il mIoU unprompted è ottimista; **l'AP50 è il numero unprompted onesto** — caveat da
  ripetere ovunque (slide comprese).
- I best checkpoint per mIoU e per AP50 cadono a epoche diverse → la metrica di selezione
  conta.

## 11. Visualizzazioni (standardizzate 2026-07-02)

Overlay 2D (`scripts/visualize_masks.py`, resi automaticamente a fine training) e viewer 3D
Gradio (`demos/demo_gradio.py`) condividono **una sola regola di selezione delle istanze**:
`train/postprocess.py::select_instances` (scarta background/score<0.5, winner-takes-all per
pixel, zero GT, nessuna assunzione sull'ordine delle query). La figura 2D ha 4 pannelli:
RGB | GT | **Predizione "onesta"** (senza GT; identica per costruzione alla colorazione del
viewer 3D) | Predizione "oracolo" (matchata con Hungarian alla GT — il limite superiore
diagnostico per distinguere "miss di detection" da "problema di qualità della maschera").
Tutte le maschere sono renderizzate alla risoluzione nativa della griglia di patch con
upsampling nearest, così GT e predizioni condividono la stessa (onesta) nitidezza — le
predizioni NON vengono lisciate bilinearmente per sembrare migliori della loro supervisione
37×37.

## 12. Stato attuale e prossimi passi

**Configurazione base attuale:** query apprese (arm C) su GT per-istanza —
`--query_mode learned --num_learned_queries 64 --instance_level`, N=190 scene di train, val =
scene0080–0089. Numeri di riferimento: **val mIoU 0.371, AP50 onesto 0.228**.

**Chiuso:** M1 (prototipo), M2 (training regolarizzato/unprompted), M3 quasi tutto (curva di
scaling → plateau; arm C → vittoria e nuova base; fix B/D → funzionano ma non vincono, bracci
chiusi; pixel decoder → neutro; sweep soglia → negativo).

**Aperto (Fase 6, ablazioni ora sensate con 200 scene e segnale di val > rumore):**
1. **Sweep del no-object weight (0.05 / 0.1 / 0.4)** — la leva indiziata per
   l'over-prediction/duplicati di arm C. È il prossimo esperimento in ordine di priorità.
2. Ablation dell'augmentation: `bundles_per_scene` 1 vs 4, `query_jitter` on/off,
   `color_jitter` on/off.
3. Densità della griglia vs recall unprompted: `--grid_size` 4/6/8.
4. Più a lungo termine: scongelamento parziale del backbone, quando il gap train−val vs N
   dirà che i dati lo sostengono.

## 13. Mappa di file, run e documenti

- **Codice del progetto:** `data/scannet_overfit.py` (loader), `models/d4rt_decoder.py`
  (query + decoder), `models/mask_upsampler.py` (pixel decoder), `train/loss.py`
  (matcher + loss), `train/eval_metrics.py` (metriche), `train/postprocess.py` (selezione
  istanze condivisa 2D/3D), `scripts/train_multiscene.py` (training vero),
  `scripts/train_overfit.py` (sanity check su singola scena), `scripts/visualize_masks.py`,
  `demos/demo_gradio.py`, `slurm/` (job + staging del dataset), `tests/` (test standalone per
  ogni componente, su CPU senza pesi del backbone).
- **Run/checkpoint:** `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/` —
  `checkpoint_best.pth` (best val mIoU, quello da usare per eval/demo),
  `checkpoint_best_ap50.pth`, `metrics.jsonl`. I checkpoint sono autosufficienti (pesi testa +
  `head_config` + bundle di scena + optimizer/scheduler); il backbone si riscarica da HF
  (`facebook/VGGT-1B`), mai salvato.
- **Documentazione:** `docs/MILESTONES.md` (riassunto consolidato — la fonte primaria),
  `docs/todo.md` (aperto), `docs/HOOK_PLAN.md` (analisi dell'hook), `CLAUDE.md` (comandi,
  storage, vincoli), `docs/old/` (dettaglio per milestone, piani eseguiti, analisi dei run di
  scaling, feedback del supervisore, brief originale, prompt del preprocessing SAM3).
