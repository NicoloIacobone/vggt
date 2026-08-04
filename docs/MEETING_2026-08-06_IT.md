# Riassunto per il meeting con i supervisori — giovedì 6 agosto 2026

*Documento di preparazione, in italiano. Ordine: dal generale al particolare. Ogni scelta è
accompagnata dal "perché". Le fonti nel repo sono citate a fine di ogni sezione, così ogni
numero è verificabile.*

---

## 0. Il messaggio in 60 secondi

1. **Cosa facciamo.** Segmentazione di istanze 3D multi-view-consistente, con un decoder
   allenato sopra un backbone **VGGT-1B strettamente congelato** (mai toccato, nemmeno con LoRA).
   Supervisione: le annotazioni 2D ufficiali di ScanNet v2.
2. **Cosa è cambiato nelle ultime due settimane.** Abbiamo sostituito la vecchia testa
   fatta in casa (famiglia "D4RT", arms A–E) con un **porting fedele del decoder MaskDINO**, e il
   salto è grosso: **+48 % mIoU e +138 % AP50** sul protocollo per-frame, **2.6× AP50** sul
   protocollo multi-view.
3. **Il modello è diventato multi-frame**: un set di query condiviso fra le viste di un bundle, con
   un blocco di *cross-frame attention*. Abbiamo costruito una metrica apposta e dimostrato che
   quel blocco serve a **preservare l'identità** dell'oggetto fra le viste, non a riconoscerlo
   meglio (`id_switch` 0.498 → 0.682 se lo si rimuove).
4. **Ora abbiamo un numero confrontabile con la letteratura.** Sul **benchmark 3D ufficiale di
   ScanNet**, split ufficiale 1201/312, valutatore ufficiale: **AP 0.023 / AP50 0.067 /
   AP25 0.268**. Siamo nell'ordine di grandezza di **FAST3DIS** (0.038 / 0.096 / 0.316) e di
   **IGGT** (0.028 / 0.112 / 0.287) pur avendo il backbone **congelato** contro i loro adattati
   con LoRA. **SegVGGT** (0.504 / 0.717 / 0.870) sta molto più in alto, ma **è un protocollo
   diverso, non un altro campionato**: il loro valutatore porta le maschere sulla nuvola di punti
   usando **pose, intrinseci e depth del sensore GT di ScanNet**, quindi misura la qualità delle
   maschere 2D *senza alcun errore di geometria*, mentre il nostro numero misura la qualità delle
   maschere 2D **moltiplicata** per la qualità della geometria predetta. Va detto chiaramente — e
   va detto altrettanto chiaramente che la loro non è una scorrettezza (§8.6).
5. **Due diagnosi solide, entrambe misurate e non ipotizzate.** (a) Il progetto è
   **data-limited**, non architecture-limited. (b) Sul righello 3D il collo di bottiglia oggi è il
   **lifting 2D→3D** (registrazione + copertura), non il decoder.
6. **Su COCO 2017** abbiamo fatto due cose diverse che non vanno confuse: una **prova di
   correttezza del porting** (riproduciamo il numero pubblicato di MaskDINO a **0.004 AP**) e uno
   **studio di backbone-swap** (VGGT congelato 37.7 AP vs DINOv2 congelato 38.8 vs ResNet-50
   congelato 34.3).

---

## 1. Il problema: cosa stiamo costruendo, e perché così

### 1.1 L'obiettivo

Dato un insieme di immagini RGB di una scena (senza pose, senza depth, senza point cloud in
ingresso), produrre **maschere di istanza coerenti fra le viste**: la stessa sedia deve essere
"la stessa istanza" in tutte le immagini in cui compare, e quelle maschere devono poter essere
sollevate in 3D.

VGGT è un modello feed-forward che, da un set di immagini, ricostruisce geometria (depth,
pointmap, camere) in un colpo solo. La nostra tesi è che **le sue feature interne contengano già
l'identità degli oggetti fra le viste** e che basti una testa leggera per estrarla.

### 1.2 I tre vincoli che ci siamo dati (e che i supervisori chiederanno di giustificare)

| Vincolo | Perché |
|---|---|
| **Backbone congelato** (VGGT non viene mai finetunato, nemmeno LoRA) | (a) È il **differenziatore** rispetto a tutti i competitor diretti: SegVGGT adatta VGGT con LoRA, FAST3DIS adatta Depth-Anything-V3 con LoRA. (b) Rende il training economico: le feature si calcolano **una volta sola per scena** e si mettono in cache, quindi un esperimento dura ore, non giorni — ed è per questo che abbiamo potuto fare decine di ablation. (c) È una domanda scientifica pulita: *quanto già c'è dentro VGGT?* |
| **Supervisione 2D** (annotazioni di istanza per frame, 19 classi) | Non usiamo mai GT 3D in training. La coerenza multi-view non è supervisionata: deve **emergere** dall'architettura. È esattamente ciò che vogliamo dimostrare. |
| **Feed-forward, nessuna ottimizzazione per scena** | La famiglia NeRF/Gaussian-Splatting ottiene la coerenza multi-view "per costruzione", ma richiede minuti di ottimizzazione **per ogni scena**. Noi vogliamo secondi, senza depth sensor né pose. |

**Conseguenza da tenere presente:** ogni volta che i nostri numeri 3D sono più bassi dei
competitor, una parte della differenza è il prezzo *voluto* di questi vincoli. Va detto, non
nascosto, ma va anche detto che il prezzo è ora **quantificato**.

*Fonti: `docs/RELATED_WORK.md`, `CLAUDE.md` §Project.*

---

## 2. La storia in due atti (perché siamo passati a MaskDINO)

### Atto I — le "arms" A–E (fino al 22 luglio, ora in `legacy/d4rt/`)

Una testa DETR-style scritta a mano, in cui abbiamo studiato **come inizializzare le query**:

- **A** — prompt a punto (centroidi GT in training, griglia uniforme in eval onesto)
- **B** — query su griglia allenate
- **C** — query DETR apprese, libere ← **la migliore**
- **D** — ibrido C+A
- **E** — query ancorate in 3D (FPS sulla pointmap predetta da VGGT)

**Esito:** C vince a ogni scala (0.367 mIoU / 0.199 AP50 multi-view a N=190), e — dato chiave —
**peggiorava aumentando i dati** (0.367 @190 → 0.350 @490). All'epoca l'abbiamo letta come "il
dataset non è il collo di bottiglia".

### Atto II — MaskDINO (dal 27 luglio, il modello attivo)

Su richiesta dei supervisori abbiamo portato il decoder **MaskDINO** (Mask2Former + DINO) sopra
lo stesso backbone congelato. Ipotesi: le arms non fallivano sulle maschere, fallivano sulla
**detection** (trovare e separare gli oggetti) — e anchor box + raffinamento iterativo +
denoising + deep supervision sono esattamente la macchina che risolve la detection nei DETR.

**Esito:** l'ipotesi era giusta sulla *classe* di architettura, e ha **ribaltato** la conclusione
sui dati: con gli stessi dati MaskDINO guadagna **+0.26 AP50** passando da 50 a 490 scene. Quindi
la vecchia testa era **architecture-limited**, e la conclusione "i dati non servono" era una
proprietà di quella testa, non del task.

> **Questa è una delle cose più utili da raccontare in meeting:** abbiamo corretto una nostra
> conclusione precedente, con una misura controllata sugli stessi dati.

*Fonti: `docs/ARMS_SUMMARY.md`, `docs/MASKDINO.md` §7.2.*

---

## 3. Come misuriamo — i quattro "righelli" (leggere PRIMA dei numeri)

Questo è il punto in cui è più facile fare confusione, ed è la prima cosa che un supervisore
attaccherà. Abbiamo **quattro protocolli diversi**, e i numeri **non sono intercambiabili**.

| # | Righello | Cosa misura | Confrontabile con... |
|---|---|---|---|
| **1** | **2D per-frame** | ogni frame valutato da solo, media sui frame poi sulle scene | le nostre arms ri-valutate sullo stesso protocollo. **Non** con la letteratura |
| **2** | **2D per-bundle (multi-view)** | un'istanza valutata **una volta sola** contro il suo volume di maschere su 8 frame: un solo IoU sui frame concatenati | arm C (0.367 / 0.199). **Non** con la letteratura |
| **3** | **3D benchmark ufficiale ScanNet** | le maschere per-vista vengono **sollevate in 3D** con depth+camere predette da VGGT, votate per superpoint e valutate dal **valutatore ufficiale** | **SÌ**: SegVGGT, FAST3DIS, IGGT |
| **4** | **COCO val2017** | AP mask/box standard con `pycocotools` | upstream MaskDINO. **Non** è un risultato di progetto (vedi §9) |

**Perché il per-frame dà sempre numeri più alti del per-bundle:** un'istanza deve fare match solo
nei frame in cui è visibile, e una predizione che non rivendica pixel in un frame viene **scartata**
invece che contata come falso positivo. Lo stesso checkpoint di arm C legge **0.451/0.294**
per-frame e **0.367/0.199** per-bundle: stesso modello, due righelli.

**Perché la regola "predizione vuota = scartata" esiste:** una query multi-view *deve* essere vuota
nei frame dove il suo oggetto non c'è. Senza quella regola il protocollo penalizzerebbe proprio i
modelli multi-view, cioè i nostri. Mask2Former/MaskDINO ottengono lo stesso effetto includendo la
probabilità media di foreground nello score.

**Un dettaglio tecnico che genera domande:** la testa di classificazione ha **19 logit sigmoid e
nessuna colonna di background** (come in DINO). "Nessun oggetto" = tutti i logit bassi. Quindi la
soglia di detection è `max_c sigmoid(logit_c) ≥ 0.25`, non un argmax.

*Fonti: `docs/RESULTS.md` §1, `docs/MASKDINO.md` §6, `train/perframe.py`.*

---

## 4. L'architettura, dal generale al particolare

### 4.1 Vista d'insieme

```
   N immagini 518×518
        │
        ▼
 ┌───────────────────────┐
 │  VGGT-1B (CONGELATO)  │  24 blocchi, attenzione alternata frame ↔ globale
 │  Aggregator           │  → feature con informazione già multi-view
 └───────────┬───────────┘
             │  hook: aggregated_tokens_list[-1] → [B, S, P, 2048]
             │  P = 1 camera + 4 register + 37×37 patch token
             ▼
 ┌───────────────────────┐
 │  Pixel decoder        │  piramide ViTDet 37²/19²/10² + encoder MSDeformAttn (6 layer)
 │  (ALLENATO)           │  + mask_features ad alta risoluzione
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐
 │  MaskDINO decoder     │  300 query, 9 layer, two-stage, anchor DAB, denoising,
 │  (ALLENATO)           │  deep supervision, (+ cross-frame attention se multi-frame)
 └───────────┬───────────┘
             ▼
   maschere + classi per istanza  ( [B, Q, S, h, w] nel caso multi-frame )
```

Parametri allenabili: **20.5 M** (contro i ~6.5 M della vecchia testa D4RT). VGGT: **0** parametri
allenati.

### 4.2 Perché una piramide se VGGT ha una sola risoluzione

MaskDINO si aspetta 3 livelli di feature (come una ResNet/FPN). VGGT è un ViT: ha **una sola
risoluzione** (37×37 token, stride 14 a 518 px). Seguiamo la ricetta **ViTDet ("simple feature
pyramid")**, che ha dimostrato che le connessioni laterali dell'FPN non sono la fonte del
guadagno: sintetizziamo i livelli con convoluzioni stride-2 dal singolo livello.

> Se un supervisore dice "ma VGGT non è un FPN, questo non può funzionare": la risposta è ViTDet,
> ed è **misurata** — vedi lo studio COCO in §9.2, dove il ViT congelato a 37×37 token **batte** la
> ResNet-50 congelata che ha 4907 celle di encoder contro le nostre 1830.

### 4.3 I cinque ingredienti di MaskDINO e a cosa servono

| Ingrediente | A cosa serve | Costo se lo si toglie (N=190) |
|---|---|---|
| **Two-stage query selection** — i token dell'encoder vengono classificati e i top-k diventano le query iniziali | dà alle query un punto di partenza informato dall'immagine invece che casuale | −0.046 AP50 |
| **Encoder deformabile (6 layer)** | arricchisce la memoria multi-scala prima del decoder | −0.044 AP50 |
| **Denoising (DN)** — GT rumorosa come query extra | acceleratore di convergenza classico in DINO | −0.030 AP50 |
| **Mask-enhanced box init** — le maschere iniziali diventano box per inizializzare gli anchor | ancora la geometria delle query | −0.016 AP50 |
| **Anchor box DAB + raffinamento layer-per-layer + deep supervision** | il cuore della detection DETR: ogni query possiede un box che si raffina a ogni layer | (non ablati singolarmente) |

**Due letture oneste, da dire in meeting:**

1. **Nessun singolo ingrediente è decisivo** (ciascuno vale 0.02–0.05 AP50). Non possiamo dire
   "è il denoising che ha fatto la differenza".
2. **Ogni variante mutilata batte comunque arm C di ~2×.** Il merito è della **classe** di
   architettura. E soprattutto: **la scala dei dati domina tutto** — +0.26 AP50 da 50→490 scene
   contro ≤0.05 di qualunque componente.

### 4.4 Le deviazioni dall'upstream (tutte deliberate e documentate)

| Deviazione | Perché |
|---|---|
| Niente detectron2/fvcore, niente kernel CUDA compilato: `MSDeformAttn` in **PyTorch puro** (`grid_sample`) | l'ambiente del cluster non ha quelle dipendenze; in più il path puro **gira su CPU**, che è ciò che rende possibile l'intera test suite CPU-only. Costo misurato: 0.005 AP su COCO (§9.1) |
| Backbone congelato VGGT invece di ResNet-50/Swin | vincolo di progetto |
| 3 scale sintetizzate da 1 (ViTDet) | VGGT ha una sola risoluzione (§4.2) |
| 19 classi sigmoid, nessun background | fedele a DINO; richiede `score_mode="sigmoid"` nelle metriche |
| Maschere sulla griglia 37×37 di default | stessa griglia delle arms → confronto onesto. La questione risoluzione è **chiusa**, vedi §7 |
| Nessuna augmentation LSJ/crop/flip | teniamo solo il jitter fotometrico del progetto, per non introdurre variabili |

*Fonti: `docs/MASKDINO.md` §1–§5, §7.2.1.*

---

## 5. Il passaggio multi-frame (il cuore del contributo)

Fino al 28 luglio il decoder vedeva **un frame alla volta**. Il passaggio multi-frame è in tre
mosse, ognuna motivata:

### Mossa 1 — `--feature_mode bundle`: feature multi-view

Si fa girare l'aggregator **una volta su tutte le S viste** invece che una volta per frame, così
l'attenzione globale di VGGT rende i token di ogni frame consapevoli delle altre viste. Nessun
parametro nuovo, cambia solo come si costruisce la cache.

**Risultato: −0.048 AP50 per-frame.** È un **risultato negativo** se guardato da solo — e va
riportato come tale. Ma è il *controllo corretto* per il decoder multi-frame, non il "bar".

### Mossa 2 — `--multi_frame`: un set di query per bundle

La query *q* è la **stessa ipotesi di oggetto in tutte le S viste**: possiede un *volume* di
maschere, non S maschere scollegate. Tre cambiamenti:

1. **Inizializzazione condivisa**: il top-k del two-stage si prende sull'**unione** dei frame e il
   contenuto viene trasmesso a tutte le viste. *Il contenuto è condiviso, la geometria (l'anchor
   box) resta per-vista*: motivazione — lo stesso oggetto ha semantica unica ma posizione diversa
   in ogni immagine.
2. **`CrossFrameAttention`**, un blocco per layer di decoder: attenzione fra le S copie della
   stessa query. **Deliberatamente senza positional encoding sui frame**, perché le viste sono un
   *insieme non ordinato* — il blocco è permutation-equivariant in S. (Conseguenza pratica
   preziosa: al momento dell'inferenza 3D funziona con qualunque numero di viste, e infatti gira
   con bundle da 3 a 55 frame pur essendo allenato a S=8.)
3. **Matching a livello di bundle**: l'assegnazione ungherese si fa **una volta per bundle** sul
   volume `[S·h·w]`, poi si proietta sui frame dove l'istanza è realmente visibile. Nelle viste
   dove non è visibile, la query è supervisionata come "nessun oggetto" — esattamente il
   comportamento che il protocollo premia.

**Costo zero per la GT:** il dataset già memorizza l'id globale di istanza per ogni target
per-frame; è il collegamento cross-view che il protocollo single-frame buttava via.

### Mossa 3 — le due ablation che localizzano il risultato (N=490)

| Config | per-frame mIoU / AP50 | **per-bundle** mIoU / AP50 | Δ bundle AP50 |
|---|---|---|---|
| multi-frame completo | 0.621 / 0.630 | **0.535 / 0.494** | — |
| … senza cross-frame attention | 0.530 / 0.524 | 0.393 / 0.311 | **−0.183** |
| … con feature per-frame invece che per bundle | 0.631 / 0.627 | 0.429 / 0.347 | **−0.147** |

**Lettura:** l'attenzione globale di VGGT *scrive* la corrispondenza cross-view dentro i token
congelati; la cross-frame attention del decoder la *consuma*. E il prezzo è l'accuratezza
per-frame: **0.729 (miglior single-frame) vs 0.630 (miglior multi-view)**. La coerenza multi-view
**non è gratis, e adesso sappiamo quanto costa**.

*Fonti: `docs/MASKDINO.md` §8.1–8.2, §7.4.1.*

---

## 6. Risultati 2D

### 6.1 La curva di scaling (split di progetto, val = scene 0080–0089)

| Modello | Scene di train | val mIoU | val AP50 | val AP75 | val mAP |
|---|---|---|---|---|---|
| **arm C — il riferimento** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO | 490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + 2 estrazioni di viste/scena + colour jitter** | **490** | **0.694** | **0.729** | **0.582** | **0.526** |

- **+48 % mIoU, +138 % AP50** sul miglior head precedente, stesso protocollo, stesso codice di
  metrica, stessa GT (arm C è stato **ri-valutato** sul protocollo per-frame, non citato dal suo
  vecchio righello — questo è importante da sottolineare).
- La curva **sale ancora** a 490 scene e l'overfitting **cala** con la scala (train mIoU
  1.000 → 0.994 → 0.947): siamo **data-limited**.
- **Caveat onesto da dichiarare:** il run `--bundles_per_scene 2` ha avuto un budget di step 2×
  per un bug di clamp delle epoche (poi corretto); ha però raggiunto il picco a 18.6k step contro
  i 15.2k del riferimento, quindi il guadagno non è semplicemente "allenato più a lungo".
- `--bundles_per_scene 4` **satura** (0.722 vs 0.729, dentro il rumore): la leva "più viste per
  scena" è esaurita a 2, la leva rimasta è **più scene**.

### 6.2 Protocollo multi-view (il righello delle arms)

| Modello (N=490) | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| arm C (N=190) | 0.367 | 0.199 | — | — |
| MaskDINO `--multi_frame` | 0.535 | 0.494 | 0.279 | 0.272 |
| **… + migliore ricetta dati** | **0.539** | **0.515** | — | — |

**+47 % mIoU, 2.6× AP50**, senza alcun matching post-hoc, tracking o fusione di maschere: la
coerenza viene dal set di query condiviso.

### 6.3 Lo split ufficiale 1201/312 (il righello onesto, 1–2 agosto)

Fino al 30 luglio usavamo uno split di progetto (val = scene 0080–0089), tenuto per continuità
della curva di scaling. Adesso abbiamo costruito e allenato sullo **split ufficiale ScanNet v2**:
1201 scene di train, 312 di val — **lo stesso su cui allenano tutti i competitor**.

| Run | protocollo | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| single-frame | per-frame (griglia 37²) | **0.624** | **0.662** | 0.487 | 0.459 |
| 〃 | per-frame (piena, 518²) | 0.611 | 0.651 | 0.466 | 0.437 |
| multi-frame | per-frame | 0.623 | 0.650 | 0.470 | 0.443 |
| 〃 | **per-bundle (multi-view)** | **0.529** | **0.525** | 0.312 | 0.311 |
| … senza cross-frame attention | per-bundle | 0.471 | 0.389 | 0.220 | — |

- **Lo split ufficiale è più difficile** (~0.07 AP50 in meno del nostro): va detto, e va detto
  *perché* (scene di val diverse, più varietà).
- **Il risultato multi-view si trasferisce**: 0.529/0.525 sullo split ufficiale contro 0.539/0.515
  sul vecchio. Non è un artefatto del nostro split.
- Il gap train/val a epoca 12 (train AP50 0.878 vs val 0.662) conferma: **ancora data-limited**.

### 6.4 La metrica di consistenza cross-view — e il risultato di meccanismo

**Problema:** `bundle_AP50` **non basta** a dimostrare la coerenza multi-view. Il *volume* di
maschere di una query può assomigliare in media a un'istanza GT anche se in ogni singola vista è
una query *diversa* a spiegare davvero l'oggetto (un "hand-off"). Abbiamo quindi costruito due
metriche (`train/eval_metrics.py::multiview_consistency_metrics`):

- **`bundle_view_consistency`** — per ogni istanza GT accoppiata, la frazione delle viste in cui è
  visibile che vengono spiegate a IoU ≥ 0.5 **dalla sua stessa query**. 1.0 = perfetto.
- **`bundle_id_switch`** — la frazione di viste in cui **un'altra** query fa meglio. 0.0 = perfetto.

Le due sono complementari per costruzione: un **miss** abbassa la prima e lascia stare la seconda;
un **hand-off** muove entrambe.

**Prime misure (split ufficiale, multi-frame):** consistency **0.717**, id_switch **0.498**, ~14.1
istanze accoppiate per bundle — ed entrambe migliorano in modo monotono durante il training
(0.679→0.717 e 0.607→0.498 dall'epoca 6 alla 12).

**Il risultato che vale come claim di meccanismo (3 agosto):** togliendo la cross-frame attention,

| | con il blocco | senza | Δ |
|---|---|---|---|
| istanze accoppiate per bundle | 14.1 | 14.0 | ±0 |
| `bundle_view_consistency` ↑ | 0.717 | 0.692 | −0.025 |
| **`bundle_id_switch`** ↓ | **0.498** | **0.682** | **+0.184** |
| per-bundle AP50 | 0.525 | 0.389 | −0.136 |

> **Il modello trova lo stesso numero di oggetti; quello che si rompe è *chi* possiede l'oggetto.**
> Il riconoscimento sopravvive, l'identità cross-view no. Questo è esattamente il meccanismo per
> cui la metrica è stata costruita, e sarà una tabella centrale del paper.

*Fonti: `docs/RESULTS.md` §2/§3/§6, `docs/MASKDINO.md` §6.6, §7.4.1, §7.8, §7.8.1.*

---

## 7. La questione risoluzione — chiusa, con una misura

**La domanda (legittima):** "predite maschere su una griglia 37×37, non è troppo grossolana?"

**La risposta è una misura, non un'opinione.** Abbiamo costruito un **oracolo**: si prende la GT
perfetta, la si quantizza sulla griglia su cui il modello predice, la si ri-upsampla e la si
valuta a piena risoluzione. Il risultato è un **tetto massimo** per quella griglia — nessun
modello, nemmeno perfetto, può superarlo.

**Su ScanNet:**

| griglia di predizione | mIoU | AP50 |
|---|---|---|
| **37×37** (nativa) | 0.910 | **0.956** |
| 74×74 | 0.963 | 0.992 |
| 148×148 | 0.983 | 0.997 |

Il modello sta a **~0.69 AP50** contro un tetto di **0.956**. **Vincola il riconoscimento, non la
risoluzione.** Confermato da due run indipendenti: `--mask_upsample 2` (74×74) resta neutro anche
sul righello a piena risoluzione (0.680 vs 0.673). Quindi il lavoro sulla risoluzione delle
maschere su ScanNet è **de-prioritizzato**, e questo libera tempo per il resto.

**Attenzione — su COCO la conclusione si ribalta** ed è per questo che lo studio COCO ha una
configurazione diversa (§9.3). Non è una contraddizione: è un effetto di *regime*. Gli oggetti di
ScanNet sono mobili che riempiono l'inquadratura; su COCO l'istanza mediana copre **8.4 celle** di
una griglia 37×37 e il 15.7 % delle istanze sta dentro **una sola cella**.

*Fonti: `docs/MASKDINO.md` §6.5, §7.7; `docs/MASKDINO_COCO.md` §1.*

---

## 8. Il righello 3D: il numero confrontabile con la letteratura

### 8.1 Perché esiste

Tutto ciò che sta sopra è misurato **da noi, sul nostro codice di metrica, su maschere 2D**.
Nessuno di quei numeri può stare accanto a SegVGGT o FAST3DIS. Quindi abbiamo costruito il loro
protocollo, per intero.

### 8.2 Il protocollo, passo per passo

1. **Un forward per scena** su tutti i frame campionati (~16–25, dall'export ufficiale
   `scannet_frames_25k`), con **un solo set di query per l'intera scena**.
2. **Pixel → query**: uno score per query = max sigmoid sulle viste; le query classificate
   wall/floor vengono scartate (non sono classi del benchmark); ogni pixel va alla sua query argmax.
3. **Unprojection + registrazione**: i pixel vengono proiettati in 3D con **la depth e le camere
   predette da VGGT** — *nessuna geometria GT entra nell'inferenza*, che è tutto il nostro punto
   di vendita. Per poter *valutare*, serve però portare la nuvola nel sistema di riferimento della
   mesh: **Sim(3) in sola valutazione** (Umeyama sui centri camera + ICP di similarità). È la
   convenzione di FAST3DIS.
4. **Lifting**: ogni punto vota per la sua query sul vertice di mesh più vicino entro
   `--vote_radius`; ogni superpoint va per intero alla query di maggioranza. Il **voto di
   maggioranza per superpoint è la convenzione di SegVGGT**; il raggio è nostro e serve solo
   perché i nostri punti cadono *vicino* alla mesh, non sopra (§8.6: nel loro protocollo non c'è
   nulla da colmare).
5. **Score con il valutatore ufficiale**, portato in Python 3 **riga per riga** e verificato in tre
   modi — incluso il test più forte: dando in pasto la **GT vera come predizione** si ottiene
   esattamente **1.000 / 1.000 / 1.000**.

### 8.3 I numeri

**Da leggere prima della tabella:** i numeri 3D pubblicati sono **due protocolli, non uno**, e la
differenza sta in *come una maschera 2D finita arriva sulla nuvola di punti del benchmark* (§8.6).

| Metodo | backbone | protocollo | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| FAST3DIS (pubblicato) | Depth-Anything-V3, **LoRA** | senza pose | 0.038 | 0.096 | 0.316 |
| IGGT (pubblicato) | — | senza pose | 0.028 | 0.112 | 0.287 |
| **noi (headline)** | **VGGT, congelato** | **senza pose** | **0.023** | **0.067** | **0.268** |
| noi, con i due knob di lifting tarati | 〃 | senza pose | 0.029 | 0.083 | 0.305 |
| SegVGGT (pubblicato) | VGGT, **LoRA** | **con pose** — non confrontabile riga-a-riga | 0.504 | 0.717 | 0.870 |

**Come presentarlo, onestamente e senza sminuirsi:**

- *"Fra i metodi valutati come noi siamo nell'ordine di grandezza di FAST3DIS (AP25 0.305 vs
  0.316) e di IGGT, senza mai toccare il backbone, mentre loro adattano il proprio con LoRA."*
- *"SegVGGT sta molto più in alto, ma è un altro protocollo: il loro valutatore trasferisce le
  maschere sulla nuvola con pose, intrinseci e depth del sensore GT di ScanNet, quindi il ponte
  2D→3D è esatto per costruzione. Il loro numero misura la qualità delle maschere 2D; il nostro
  misura la qualità delle maschere 2D per la qualità della geometria predetta."*
- *"Non è una scorrettezza da parte loro: il loro modello prende immagini senza pose esattamente
  come il nostro, e isolare la segmentazione dalla ricostruzione è una scelta di valutazione
  legittima. Il problema è solo che in letteratura i due protocolli finiscono nella stessa
  tabella senza distinzione."*
- *"La riga headline è quella con i knob di default, perché i knob della seconda riga erano stati
  scelti su un run diagnostico che includeva scene di val."*

### 8.4 Le due scoperte del righello 3D

**(a) Il checkpoint senza leakage BATTE quello con leakage.** Un checkpoint allenato su scene che
**includevano** le scene di val faceva 0.052 AP50; quello allenato correttamente su 1201 scene
ufficiali (val mai vista) fa **0.083** — 1.6× meglio *nonostante lo svantaggio*. Cioè: **la scala
dei dati batte il vedere le scene di test**. È il righello 3D che riproduce, indipendentemente, la
conclusione "data-limited" ottenuta in 2D.

**(b) Il collo di bottiglia oggi è il lifting, non il decoder.** Tre indizi convergenti:

- **AP25 ≈ 4× AP50**: gli oggetti vengono trovati e collocati grossolanamente, ma le maschere
  sollevate non superano la soglia IoU 0.5.
- **La registrazione ha un errore mediano di 0.14 m** sui centri camera (≈ il raggio di voto).
- **Solo ~16 % dei vertici della mesh riceve un voto**; ~65 % dei vertici annotati viene assegnato.
  Ogni istanza non coperta è un falso negativo secco.

E una **sweep pulita di 8 punti** sui due knob di lifting mostra che: il raggio di voto **satura a
~0.15 m — esattamente la scala dell'errore di registrazione** (raddoppiarlo oltre non cambia
niente), e l'intera griglia va da 0.067 a 0.091 AP50, **restando comunque sotto** lo 0.096 di
FAST3DIS. **Utile negativo:** il gap residuo *non* è un artefatto di tuning, viene da copertura e
qualità di registrazione.

### 8.5 Due handicap strutturali da dichiarare quando si legge la tabella

1. **`otherfurniture`**: è 1 delle 18 classi del benchmark, ma nella nostra GT 2D è background —
   la nostra testa a 19 classi non può predirla. Riportiamo anche una media a 17 classi come
   diagnostica (0.024 / 0.071 / 0.284).
2. **Copertura dei frame**: siamo limitati ai ~16–25 frame per scena dell'export ufficiale. Il
   "2–24 frame" di SegVGGT è il loro campionamento in *training*: in **valutazione** prendono un
   frame ogni 20 di un'estrazione `.sens` completa, cioè ~75–100 viste per scena. Su questo non
   siamo confrontabili con loro, siamo indietro.

### 8.6 I due protocolli 3D — la cosa da sapere prima di confrontarsi con SegVGGT

Verificato il 2026-08-04 leggendo **il loro codice di valutazione rilasciato**
(clone in `/cluster/scratch/niacobone/SegVGGT`), non l'articolo:

| | **con pose** (posed transfer) | **senza pose / geometria predetta** |
|---|---|---|
| chi | **SegVGGT** 0.504 / 0.717 / 0.870 | **FAST3DIS** 0.038 / 0.096 / 0.316, **IGGT** 0.028 / 0.112 / 0.287, **noi** 0.023 / 0.067 / 0.268 |
| come le maschere arrivano sulla nuvola | la nuvola GT viene **proiettata dentro ogni vista** con pose e intrinseci **GT** di ScanNet; l'occlusione si risolve con la **depth del sensore** | i pixel vengono **sollevati in 3D** con depth e camere **predette** dal modello, poi Sim(3)+ICP per poter valutare |
| errore di geometria nel ponte | **zero**, la corrispondenza 3D↔2D è esatta per costruzione | tutto l'errore della ricostruzione feed-forward (da noi: 0.14 m mediani sui centri camera) |
| cosa misura quindi il numero | qualità delle maschere 2D | qualità delle maschere 2D **×** qualità della geometria predetta |
| valutatore | ufficiale ScanNet, stesse opzioni | ufficiale ScanNet, stesse opzioni |

Le prove, riga per riga: `eval/eval_instance_seg.py:243-336` non fa alcuna unprojection, proietta
la nuvola GT nelle viste; le pose vengono da `pose/{frame}.txt` (`:198`), gli intrinseci da
`intrinsic_depth.txt` (`eval/instance_eval_common.py:68`), l'occlusione dalla depth del sensore
`depth/{frame}.png` entro 0.1 m (`:178-182`, `:305-307`, `:451`); di conseguenza **niente Sim(3),
niente ICP, nessun raggio di voto**; le teste di geometria di VGGT non vengono **mai** chiamate
(`instance_eval_common.py:168-189` usa solo l'aggregator e la testa semantica); e il codice di
metrica è la copia mmdet3d del valutatore ufficiale ScanNet **con le nostre stesse opzioni**
(`eval/instance_seg_eval.py:523-540` vs il nostro `train/benchmark3d.py:36-37`). **Il valutatore
non è la differenza: la differenza è il ponte 2D→3D.**

Differenze secondarie, tutte a loro favore ma nessuna decisiva: ~75–100 viste per scena contro le
nostre ~17; maschere a 259×196 contro la nostra griglia 37×37; 600 predizioni tenute contro le
nostre 100; e loro allenano e vengono valutati anche su `otherfurniture`, che la nostra testa non
può predire.

**Da dire con equilibrio, se esce in riunione:** SegVGGT non sta barando e non va raccontato così.
Il loro **modello** prende solo immagini senza pose, esattamente come il nostro; la geometria GT
serve unicamente a trasferire maschere già fatte sulla nuvola per il calcolo del punteggio, il che
isola deliberatamente la qualità di segmentazione da quella di ricostruzione — una scelta di
valutazione legittima. Il problema è solo che in letteratura i due protocolli compaiono nella
stessa tabella senza distinzione; e SegVGGT e FAST3DIS sono preprint contemporanei
(2603.19926 e 2603.25993), quindi nessuno dei due poteva citare l'altro.

*Fonti: `docs/MASKDINO.md` §9 (protocollo, §9.6 il numero, §9.8 la sweep, §9.9 i due protocolli),
`docs/RESULTS.md` §5, `docs/RELATED_WORK.md`.*

---

## 9. COCO 2017 — due esperimenti diversi, da non confondere

Questa è la parte che rischia di più di essere fraintesa in meeting, quindi va introdotta così:
**"Su COCO abbiamo fatto due cose che rispondono a due domande diverse."**

### 9.1 Esperimento A — il "trapianto": la nostra implementazione è fedele?

**La domanda.** Tutti i numeri su ScanNet confrontano il nostro porting con le *nostre* baseline.
Se avessimo un bug che è "sbagliato allo stesso modo" da entrambe le parti, non lo vedremmo mai.

**Il metodo.** Carichiamo i **pesi COCO ufficiali rilasciati da MaskDINO** nell'harness detectron2
di upstream, poi **sostituiamo a caldo il nostro decoder e il nostro encoder deformabile** e
rifacciamo l'intera valutazione su val2017. `--mode baseline` (upstream intatto) è il controllo.

**Il fatto tecnico più forte:** il nostro decoder accetta i pesi di upstream a `strict=True` —
**333/333 parametri**, nomi e shape.

| COCO val2017, 5000 immagini | segm AP | segm AP50 | box AP | box AP50 |
|---|---|---|---|---|
| model zoo di upstream, questo esatto checkpoint | 46.1 | — | 51.5 | — |
| `--mode baseline` (codice upstream, nostro ambiente) | 46.129 | 69.021 | 51.540 | 70.509 |
| **`--mode ours` (nostri moduli)** | **46.133** | 69.036 | **51.549** | 70.514 |

**Δ = +0.004 segm AP / +0.009 box AP.** Su CPU i due modi sono **bit-identici**; la deriva di
~0.005 compare solo su GPU perché upstream chiama il kernel CUDA fuso mentre noi usiamo sempre il
core `grid_sample` portabile. È l'unica differenza voluta fra le due implementazioni.

**Verificato che non sia un no-op** (domanda da supervisore quasi certa): il codice *asserisce a
runtime* che i moduli in esecuzione sono i nostri (6 encoder + 9 decoder deformabili), e
perturbando un singolo peso **dentro il nostro decoder** di 1.05× lo score si muove
(55.702 → 55.608 su un subset). Quindi numeri identici significano **equivalenza**, non fallback
silenzioso.

**Cosa certifica e cosa no** (da dire prima che lo chiedano):

| Certificato | Non esercitato da questa via |
|---|---|
| attenzione deformabile (encoder e decoder), stack dell'encoder e reference point, two-stage, anchor DAB, raffinamento iterativo dei box, mask-enhanced box init, teste di predizione | matcher, criterion (solo training), generazione delle query di denoising, modulo multi-frame (non ha controparte upstream), la piramide ViTDet su VGGT (non ha controparte COCO) |

La colonna di destra non è un problema noto: è coperta dagli unit test (loss a zero su predizioni
perfette, overfit sintetici).

**Non va MAI messo accanto a un numero ScanNet.** Lì il backbone, il dataset e il task sono tutti
di upstream. Dice che la nostra implementazione di MaskDINO è fedele, e nient'altro.

### 9.2 Esperimento B — il backbone-swap: quanto costa mettere VGGT al posto della ResNet?

**La domanda.** L'esperimento A non tocca mai il **training** e non dice niente sul **backbone**.
Qui alleniamo **lo stesso decoder** su COCO instance segmentation, su feature **congelate**,
cambiando solo il backbone. Tre bracci, tutto il resto identico (schedule, dati, augmentation,
loss, risoluzione della GT):

| braccio | backbone | celle di encoder @518 px | `mask_features` | tetto di risoluzione |
|---|---|---|---|---|
| `resnet50` | ImageNet R50, congelata | 65²/33²/17² = 4907 | stride 4 → 130² | ~92 |
| `vggt` | VGGT-1B aggregator, congelato | 37²/19²/10² = 1830 | deconv ×4 → 148² | 84.2 |
| `dinov2` | DINOv2 ViT-L/14-reg, congelato | 37²/19²/10² (identiche a VGGT) | deconv ×4 → 148² | 84.2 |

**Perché proprio questi tre.** Ogni confronto compra una cosa precisa:

- **`vggt` vs `resnet50`** — il titolo: quanto costa lo swap, tutto il resto fisso. È confuso dal
  numero di token (1830 vs 4907): lo dichiariamo, non lo nascondiamo.
- **`vggt` vs `dinov2`** — il confronto **pulito**: stessa patch size, stessa famiglia, stesso
  numero di token. DINOv2 ViT-L/14-reg è *esattamente* il modello da cui è costruito il
  `patch_embed` di VGGT (il checkpoint ufficiale entra nel `vit_large` vendorizzato di VGGT a
  `strict=True`, 0 chiavi mancanti). Quindi questo confronto isola **cosa ha fatto il pretraining
  3D di VGGT alla semantica 2D**, separandolo da **quanto costa la griglia 37×37**.
- **`resnet50` congelata a 12 epoche vs upstream finetunata a 50 epoche (46.1)** — la distanza
  onesta dal numero pubblicato.

**Risultati (tutti e tre completi, 1 agosto, val2017 completo, step 87 948 = 12 epoche):**

| braccio | segm AP | AP50 | AP75 | APs | APm | APl | box AP | tetto | miglior checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| upstream R50, finetunata, 50 ep (pubblicato) | 46.1 | — | — | — | — | — | 51.5 | 92.0 | — |
| `resnet50` congelata, 12 ep | 34.3 | 54.1 | 36.2 | 14.3 | 36.1 | 53.6 | 38.2 | ~92 | 36.7 |
| `vggt` congelato, 12 ep | 37.7 | 59.4 | 39.5 | 15.3 | 41.6 | 58.5 | 42.1 | 84.2 | 39.7 |
| `dinov2` congelato, 12 ep | **38.8** | 64.8 | 39.6 | 14.8 | 43.0 | 65.1 | 45.9 | 84.2 | **41.3** |

**Tre letture:**

1. **La griglia 37×37 (+ mask_upsample 4) non è invalidante: vince.** DINOv2 congelato con 1830
   celle batte la ResNet-50 congelata con 4907 celle di **+4.4 AP**, e anche il suo `APs` (oggetti
   piccoli, 14.8) pareggia quello della R50 (14.3). La preoccupazione "un ViT a bassa densità di
   token non può fare instance segmentation" **non si materializza** come deficit contro il
   controllo.
2. **Congelare + 12 epoche costa alla ricetta R50 circa 12 AP** rispetto al 46.1 finetunato a 50
   epoche. È la distanza onesta dal numero pubblicato, misurata apposta.
3. **Il pretraining 3D di VGGT costa ~1–1.6 AP di semantica 2D a geometria di token identica**
   (37.7 vs 38.8 finale; 39.7 vs 41.3 sul miglior checkpoint). I due bracci partono lontanissimi
   (14.1 vs 23.4 al gate di overfit), convergono a metà training e divergono nel finale: `vggt` va
   in plateau a 75k step mentre `dinov2` continua a salire fino a 85k. **È shift di dominio, non
   scarsità di token** — entrambi battono comunque la R50 congelata.

### 9.3 La misura che ha reso l'esperimento B possibile: il tetto di risoluzione su COCO

Prima di spendere ~90 ore GPU abbiamo misurato il **tetto** di ogni griglia con lo stesso metodo
oracolo di §7, ma su COCO:

| griglia di predizione | AP | **APs** |
|---|---|---|
| **37×37** (la griglia della pista ScanNet) | **44.7** | **8.0** |
| 74×74 | 66.2 | 38.4 |
| **148×148 (`--mask_upsample 4`)** | **84.2** | 70.1 |
| stride 4 @800 — quella di MaskDINO-su-R50 | 92.0 | 84.8 |

**Su una griglia 37×37 un modello perfetto fa 44.7 — sotto il 46.1 che sarebbe il target.**
L'esperimento sarebbe stato **irrisolvibile per costruzione**. Da qui due conclusioni operative:

- **Problema A, la griglia delle maschere: reale ma economico.** La risoluzione di `mask_features`
  **non è legata** a quella dei token: viene da conv trasposte, come in ViTDet (un ViT stride-16
  che alimenta una mask head stride-4). Con `--mask_upsample 4` si ottengono maschere 148×148 e un
  tetto di 84.2. Costo: due `ConvTranspose2d`. **È il default dei bracci ViT su COCO, ed è la
  differenza più importante rispetto alla ricetta ScanNet.**
- **Problema B, la griglia dei token: reale, costoso, e NON risolto qui.** 1369 token determinano
  quanto bene oggetti piccoli possono essere *rilevati e separati*. Un braccio VGGT a 1036 px
  (74×74 token ≈ le 65×65 della R50) costerebbe ~5× il backbone; è **rimandato, non confutato**.
  (VGGT usa RoPE 2D senza position embedding assoluto, quindi accetta qualunque griglia: l'abbiamo
  verificato.)

Un'ultima scelta motivata: **si "schiaccia" l'immagine in un quadrato, non si fa padding.** Il
padding centrato (la funzione di preprocessing di VGGT) spende celle di griglia su bordi neri e
costa **4.9 AP di tetto** a parità di token (39.8 vs 44.7).

### 9.4 Cosa si può dire e cosa no, con i numeri COCO

| Si può dire | NON si può dire |
|---|---|
| "il nostro porting di MaskDINO riproduce upstream a 0.004 AP" | "abbiamo riprodotto MaskDINO" (i bracci di §9.2 sono congelati e a 12 epoche) |
| "un backbone ViT congelato a 1830 celle batte una R50 congelata a 4907 celle" | mettere un numero COCO accanto a un numero ScanNet |
| "il pretraining 3D di VGGT costa ~1–1.6 AP di semantica 2D" | "VGGT è peggio di DINOv2" in senso assoluto (su COCO 2D sì; sul nostro task la geometria è il motivo per cui usiamo VGGT) |

*Fonti: `docs/MASKDINO.md` §7.6; `docs/MASKDINO_COCO.md` §1, §2, §6.*

---

## 10. Posizionamento rispetto allo stato dell'arte

### 10.1 Il contesto scomodo, da affrontare per primo

"Attaccare un decoder a un backbone congelato della famiglia VGGT/DUSt3R e fare un task 3D
downstream" è diventato **il pattern dominante degli ultimi 12 mesi** (il genere "VGGT-X":
VGGT-Det, VGGT-Occ, DriveVGGT, …). **L'architettura da sola non è più un contributo.**

### 10.2 Cosa è già rivendicato da altri (e quindi non rivendichiamo)

| Meccanismo | Di chi è | Cosa resta a noi |
|---|---|---|
| Query condivise fra le viste su backbone VGGT | **SegVGGT** (400 query in tutti i 24 layer dell'aggregator) | il nostro `--multi_frame` è la stessa *classe* di idea → va riportato come confronto controllato contro il nostro single-frame, non come meccanismo nuovo |
| **Query ancorate in 3D** (generatore di anchor 3D + project-and-sample) | **FAST3DIS** | il nostro §8.3 diventa un'**ablation**, non un contributo: "anchor 3D vs box DAB 2D, stesso backbone congelato, stessi dati, stesso protocollo" — che nessuno ha fatto |
| Fix della dispersione dell'attenzione su molti token | SegVGGT (FADA) | noi lo otteniamo in parte gratis dall'attenzione deformabile (4 punti campionati per livello invece che attenzione densa) |
| Predizione panottica multi-view single-pass | PanSt3R | è il nostro punto di confronto "perché non splat / perché non fondere" |
| Regolarizzazione anti-duplicato per le query | FAST3DIS (contrastiva + penalità di overlap) | noi lo risolviamo col matching ungherese uno-a-uno + DN: contrasto legittimo in discussione |

### 10.2-bis Un contributo di posizionamento che non avevamo previsto

Leggendo il codice di valutazione rilasciato da SegVGGT (§8.6) abbiamo trovato che **i numeri 3D
pubblicati stanno su due protocolli diversi presentati come uno solo**: SegVGGT trasferisce le
maschere sulla nuvola con pose, intrinseci e depth GT di ScanNet ("con pose"), mentre FAST3DIS,
IGGT e noi le trasferiamo con la geometria **predetta** ("senza pose"). Il valutatore è lo stesso,
con le stesse opzioni: la differenza è il ponte 2D→3D. È il motivo per cui FAST3DIS e IGGT si
raggruppano con noi e SegVGGT no. Non è un'accusa — la loro è una scelta legittima che isola la
segmentazione dalla ricostruzione — ma è una distinzione che in letteratura oggi non viene fatta,
e dichiararla nel nostro capitolo di valutazione è di per sé un piccolo contributo.

### 10.3 Il nostro contributo, formulato in una frase

> **Uno studio controllato**: un backbone **strettamente congelato**, un dataset, un protocollo,
> ingredienti del decoder variati **uno alla volta** — con una metrica che isola *cosa* compra la
> coerenza cross-view (l'identità, non il riconoscimento) e un numero sul benchmark 3D ufficiale
> che rende il tutto collocabile.

I tre pilastri concreti:

1. **Il backbone congelato è un differenziatore deliberato** — ogni competitor diretto usa LoRA.
2. **La tabella di ablation multi-frame** (cross-frame attention 0.183 / bundle features 0.147 di
   bundle AP50, più il prezzo in accuratezza per-frame quantificato) è la tabella centrale.
3. **Il claim di meccanismo misurato**: la cross-frame attention preserva l'identità
   (`id_switch` 0.498 → 0.682 senza), non il riconoscimento (14.1 vs 14.0 istanze trovate).

### 10.4 La risposta pronta a "perché non usate Gaussian Splatting / NeRF?"

Quei metodi ottengono la coerenza multi-view *per costruzione* (una sola rappresentazione 3D
renderizzata su tutte le viste) ma richiedono **ottimizzazione per scena**. Il nostro pitch:
feed-forward, nessuna ottimizzazione per scena, nessuna geometria GT né sensore di profondità in
input, **secondi invece di minuti**.

*Fonti: `docs/RELATED_WORK.md`.*

---

## 11. Dataset e infrastruttura — dove è finito il tempo

Vale la pena raccontarlo, perché una fetta reale delle due settimane è andata qui e spiega
"perché non ci sono più esperimenti".

- **Estensione allo split ufficiale 1201 scene.** Il primo tentativo è **fallito sulla quota di
  inode** dello scratch (1.0 M soft / 1.5 M hard **file**): l'albero di build è ~1046 file per
  scena, cioè ~1.26 M file per 1201 scene. I job sono morti a 1090/1201 scene lasciando l'account
  a 1 499 966 file su 1 500 000 — impossibile scrivere qualunque cosa. **Soluzione:** la build
  ora avviene **node-local** in `$TMPDIR` e solo un tar compresso per chunk tocca lo scratch (1
  inode). Il secondo tentativo ha **preservato** il lavoro del primo (~90 ore-nodo di streaming
  già fatte) invece di riscaricare.
- **Risultato:** `scannet_official_gt_1201.tar.zst` (29 GB, 1201 scene, 17 638 istanze, 0
  duplicati cross-classe) e `scannet_official_gt_val312.tar.zst` (7.4 GB, 312 scene, 4630
  istanze). Le due liste sono disgiunte: insieme sono il protocollo ufficiale completo.
- **Dati per il righello 3D** (1 agosto): mesh + superpoint + aggregation per le 312 scene di val,
  e i frame ufficiali `scannet_frames_25k` con le pose (5436 frame). *Nota:* i nostri vecchi
  subset a stride-5 coprivano solo i frame grezzi 0–495 e avrebbero **tagliato il recall**, quindi
  è stato necessario riscaricare l'export ufficiale.
- **Economia di training:** con le feature congelate in cache, i due run sullo split ufficiale
  sono costati **8h16** (single-frame, con anche il righello a piena risoluzione) e **5h42**
  (multi-frame) su **una sola RTX 4090**. È questo che rende possibile lo studio controllato.
- **Strumenti qualitativi** (3 agosto): il visualizzatore Gradio ora accetta i checkpoint
  MaskDINO e colora **la point cloud predetta da VGGT** con le istanze della testa; c'è un tab
  "GT vs Prediction (synced)" con **una sola camera** per i due pannelli; e `scripts/view_ply.py`
  trasforma un `.ply` in una pagina HTML autonoma per guardarlo senza MeshLab.

**Due figure diverse, da non confondere in presentazione:**
(a) `--dump_ply` mostra **ciò che il benchmark valuta** (i vertici della mesh dopo lifting e voto;
il grigio = nessuna istanza è arrivata lì → è la copertura del §8.4 resa visibile);
(b) il viewer Gradio mostra **ciò che il modello predice** (la nuvola di VGGT colorata dalla
testa, senza mesh né registrazione). Una predizione che sembra giusta in (b) e vuota in (a) **è un
problema di lifting** — che è esattamente la diagnosi numerica del §8.4.

*Fonti: `docs/DATASET.md` §5.1, `docs/todo.md` 1c/1d, `docs/MASKDINO.md` §9.3, §9.7.*

---

## 12. Cronologia delle ultime due settimane (per ricostruire cosa è successo quando)

| Data | Cosa |
|---|---|
| 22 lug | Chiusa la sweep di tutte le arms a 490 scene: **arm C vince a ogni scala**, ranking invariato |
| 27 lug | **Porting di MaskDINO** completato + test CPU verdi; curva di scaling 50/190/490 → **0.699 AP50** |
| 28 lug | Ablation single-frame (nessun ingrediente decisivo); `--feature_mode bundle` (negativo per-frame); **`--multi_frame` implementato**; nuovo best 0.729 con 2 estrazioni di viste |
| 29 lug | **Ablation multi-frame**: cross-frame attention vale 0.183 bundle AP50; **check di equivalenza su COCO** (Δ 0.004 AP); colori delle figure agganciati all'identità e non al rank |
| 30 lug | **Questione risoluzione chiusa** (oracolo ScanNet: tetto 0.956 vs modello 0.69); nuovo best multi-view **0.539 / 0.515**; **build 1201 scene riuscita** dopo il fallimento sugli inode |
| 1 ago | Build **val-312**; **bracci COCO completi** (r50/vggt/dinov2); **pipeline 3D costruita e verificata** (GT come predizione → 1.000); run diagnostici 3D; metrica di consistenza implementata; `checkpoint_best_bundle.pth` |
| 2 ago | **Primi run sullo split ufficiale 1201/312**: SF 0.624/0.662, MF per-bundle 0.529/0.525, prima misura di consistenza 0.717 |
| 3 ago | **Numero 3D riportabile** (0.023/0.067/0.268); **ablation di consistenza** (id_switch 0.498→0.682); sweep dei knob di lifting; strumenti di visualizzazione 3D |

---

## 13. Cosa manca — il piano che proporrei al meeting

Ordinato per valore atteso, con la motivazione di ciascuno:

1. **Copertura del lifting** (`todo 5b`) — solo ~16 % dei vertici riceve un voto e ~65 % dei
   vertici annotati viene assegnato: è un tetto secco sul recall. Opzioni: più frame per scena,
   bundle sovrapposti, voto pesato per confidenza invece di argmax duro. **Metrica intermedia da
   guardare: `annotated_assigned_frac`, non AP** (così non si insegue il rumore).
2. **Qualità della registrazione** (`todo 5c`) — il nostro RMS è dell'ordine del raggio di voto.
   Idee: ICP sui *punti votati* invece che sui centri camera; registrazione per-bundle con un
   controllo di consistenza. **Vincolo da non violare: la registrazione è solo in valutazione, non
   deve mai far entrare geometria GT nel path di predizione** — è il punto di vendita del progetto.
3. **Anchor 3D vs box DAB 2D** (`todo 2d`) — da fare **sopra** il multi-frame (un anchor 3D ha
   senso solo se una query è un'istanza attraverso le viste). Da presentare come **ablation**, non
   come meccanismo nuovo (FAST3DIS lo possiede), e chiude anche il cerchio sul risultato negativo
   di arm E.
4. **Solo dopo 1–3**: rivedere se il decoder torna a essere il vincolo su questo righello.

**Cosa NON faremo, e perché** (utile da dire, mostra controllo dello scope):

- **Non** lavoriamo più sulla risoluzione delle maschere su ScanNet (§7: il tetto è 0.956, il
  modello sta a 0.69).
- **Non** scongeliamo il backbone: rinuncerebbe al differenziatore. Se lo faremo, sarà una
  decisione esplicita e motivata, non un default.
- `--bundles_per_scene 4` è **saturo**: la leva "più viste per scena" è finita.

---

## 14. Domande da supervisore — con le risposte pronte

**Q: "Perché congelate il backbone? Con LoRA fareste molto meglio."**
Sì, quasi certamente. Ma è precisamente il differenziatore: **ogni** competitor diretto adatta il
proprio backbone (SegVGGT con LoRA su VGGT, FAST3DIS con LoRA su DA3). La domanda scientifica che
poniamo è *quanto è già dentro VGGT*. In più, il congelamento è ciò che rende economico lo studio
controllato (feature in cache → 6 ore per run su una GPU → decine di ablation). Se serve il numero
massimo, scongelare è la leva ovvia — ma è un'altra tesi.

**Q: "I vostri numeri 3D sono molto più bassi di SegVGGT. Il metodo funziona?"**
Prima di tutto: **quel confronto attraversa due protocolli diversi** (§8.6). Abbiamo letto il loro
codice di valutazione: SegVGGT non solleva nulla in 3D, proietta la nuvola GT del benchmark dentro
ogni vista usando **pose, intrinseci e depth del sensore GT di ScanNet**, quindi nel loro punteggio
non entra **alcun** errore di geometria. Il loro numero misura la qualità delle maschere 2D; il
nostro misura la qualità delle maschere 2D **moltiplicata** per la qualità della geometria che il
modello predice da solo. Non è una scorrettezza — il loro modello prende immagini senza pose come
il nostro, e isolare la segmentazione dalla ricostruzione è legittimo — ma le due righe non sono
confrontabili una accanto all'altra. Il confronto giusto è con chi usa il nostro protocollo:
FAST3DIS e IGGT, dove su AP25 siamo a 0.305 contro il loro 0.316 e 0.287, quindi il
*riconoscimento* funziona. La perdita è concentrata sul salto da AP25 a AP50 (fattore ~4), e le
diagnostiche dicono esattamente perché: errore di registrazione mediano 0.14 m e solo ~16 % di
vertici coperti. Abbiamo dimostrato con una sweep di 8 punti che **non è un problema di tuning**
(il raggio di voto satura proprio a 0.15 m = l'errore di registrazione). Detto ciò, SegVGGT ha
comunque due vantaggi reali che non neghiamo: adatta il backbone con LoRA e in valutazione usa
~75–100 viste per scena contro le nostre ~17.

**Q: "Come sapete che il vostro MaskDINO non ha bug?"**
Tre livelli. (1) I pesi ufficiali di upstream entrano nel nostro decoder a `strict=True`,
**333/333 parametri**. (2) Guidato da quei pesi, riproduce il numero pubblicato a **0.004 AP**,
contro un controllo eseguito nello stesso ambiente. (3) Abbiamo verificato che non sia un no-op
perturbando un peso *dentro il nostro decoder* e vedendo lo score muoversi. Il matcher e il
criterion non passano da questa via — quelli poggiano su test unitari (loss a zero su predizioni
perfette, overfit sintetici).

**Q: "37×37 token sono pochissimi per la segmentazione."**
Su ScanNet è misurato e falso: il tetto di quella griglia è **0.956 AP50** e il modello sta a 0.69.
Su COCO invece è **vero** (tetto 44.7 AP) — ed è per questo che nei bracci COCO usiamo
`--mask_upsample 4`, che porta il tetto a 84.2 con due conv trasposte. La differenza è il regime:
l'istanza mediana di COCO copre 8.4 celle, i mobili di ScanNet ne coprono molte di più.
**Nota importante:** la risoluzione delle *maschere* è disaccoppiata da quella dei *token* — la
seconda resta un limite reale e la dichiariamo (1830 celle contro 4907 della R50), ma nello studio
COCO il ViT congelato **vince comunque** di +4.4 AP.

**Q: "Il vostro split di validazione non è quello ufficiale."**
Non lo era: le scene 0080–0089 erano una convenzione di progetto, tenuta per continuità della
curva di scaling. **Dal 2 agosto alleniamo e validiamo sullo split ufficiale 1201/312**, e i
risultati si trasferiscono (multi-view 0.529/0.525 ufficiale vs 0.539/0.515 vecchio). Lo split
ufficiale è ~0.07 AP50 più difficile.

**Q: "C'è leakage nei vostri numeri?"**
Ce n'era, ed è documentato: i primi run 3D usavano un checkpoint allenato su scene che includevano
la val ufficiale — li abbiamo etichettati **diagnostici** e non li citiamo. Il numero riportabile
viene dal checkpoint allenato sulle 1201 scene ufficiali con la val mai vista. Curiosità utile: il
checkpoint **senza** leakage batte quello con leakage di 1.6×.

**Q: "Perché AP50 e AP25 e non solo AP?"**
Perché la loro differenza è **diagnostica**: AP25 alto e AP50 basso significa "oggetto trovato e
collocato, ma maschera sollevata imprecisa" — cioè un problema di geometria/lifting, non di
riconoscimento. È il ragionamento che ci ha portato ad aprire il workstream sul lifting invece di
continuare a modificare il decoder.

**Q: "Il vostro contributo qual è, se SegVGGT e FAST3DIS esistono già?"**
Lo studio controllato, non il meccanismo. Nessuno ha pubblicato: un backbone congelato, un
dataset, un protocollo, ingredienti variati uno alla volta, con una metrica che **separa
riconoscimento e identità cross-view**. Il nostro risultato di meccanismo (la cross-frame
attention preserva l'identità, non il riconoscimento) è ottenuto proprio grazie a quella
separazione, e non compare in nessuno dei due lavori.

**Q: "Quanti frame usate, e il modello generalizza a numeri diversi?"**
Alleniamo con bundle di 8 frame; in inferenza 3D usiamo tutti i frame della scena (3–55, mediana
15) **senza riaddestrare**, perché il blocco cross-frame è deliberatamente privo di positional
encoding sui frame, quindi è definito per qualunque S. Zero fallimenti su 312 scene.

---

## 15. Appendice — tabelle di riferimento rapido

### 15.1 Tutti i numeri principali su una pagina

**Righello 1 — 2D per-frame** (split di progetto, val 0080–0089)

| | mIoU | AP50 |
|---|---|---|
| arm C (miglior testa precedente) | 0.451 | 0.294 |
| MaskDINO N=490 | 0.669 | 0.699 |
| **MaskDINO N=490 + ricetta dati** | **0.694** | **0.729** |

**Righello 2 — 2D per-bundle** (stesso split)

| | mIoU | AP50 |
|---|---|---|
| arm C | 0.367 | 0.199 |
| **MaskDINO multi-frame + ricetta dati** | **0.539** | **0.515** |

**Righello 1+2 — split ufficiale 1201/312** (il righello onesto)

| | mIoU | AP50 |
|---|---|---|
| single-frame, per-frame | 0.624 | 0.662 |
| multi-frame, per-bundle | 0.529 | 0.525 |
| consistenza: `view_consistency` 0.717 / `id_switch` 0.498 | | |

**Righello 3 — benchmark 3D ufficiale** (val-312, valutatore ufficiale). Due protocolli, §8.6:

| | protocollo | AP | AP50 | AP25 |
|---|---|---|---|---|
| FAST3DIS (LoRA) | senza pose | 0.038 | 0.096 | 0.316 |
| IGGT | senza pose | 0.028 | 0.112 | 0.287 |
| **noi (congelato)** | **senza pose** | **0.023** | **0.067** | **0.268** |
| SegVGGT (LoRA) | **con pose** — non confrontabile riga-a-riga | 0.504 | 0.717 | 0.870 |

**Righello 4 — COCO val2017** (verifica, non risultato di progetto)

| | segm AP | box AP |
|---|---|---|
| upstream, checkpoint ufficiale | 46.1 | 51.5 |
| il nostro porting con gli stessi pesi | 46.133 | 51.549 |
| backbone-swap congelati, 12 ep: r50 / vggt / dinov2 | 34.3 / 37.7 / **38.8** | 38.2 / 42.1 / 45.9 |

### 15.2 Dove sta ogni cosa nel repo

| Documento | Contenuto |
|---|---|
| `docs/MASKDINO.md` | **il documento primario**: architettura, protocolli, tutti i risultati, il righello 3D (§9) |
| `docs/MASKDINO_COCO.md` | lo studio COCO di backbone-swap + la misura del tetto di risoluzione (§1) |
| `docs/RESULTS.md` | tutti i numeri, divisi per protocollo — §1 spiega perché non si mescolano |
| `docs/SUPERVISOR_COMPARISON.md` | il riassunto "da mandare fuori" in inglese |
| `docs/RELATED_WORK.md` | competitor e posizionamento |
| `docs/DATASET.md` | provenienza della GT, i tar, la lezione sugli inode |
| `docs/todo.md` | solo il lavoro aperto |
| `docs/ARMS_SUMMARY.md` | le arms A–E ritirate, in una pagina |

### 15.3 Glossario minimo

- **Bundle** — un gruppo di S frame (8 in training) della stessa scena, trattato come un'unità.
- **Query** — uno slot del decoder; nel multi-frame **una query = un'istanza in tutte le viste**.
- **Two-stage** — i token dell'encoder vengono classificati e i migliori diventano query iniziali.
- **DN (denoising)** — GT rumorosa aggiunta come query extra per accelerare la convergenza.
- **Superpoint** — segmento geometrico della mesh ScanNet; l'unità su cui si fa il voto di
  maggioranza nel lifting.
- **Sim(3)** — trasformazione di similarità (rotazione + traslazione + scala); qui usata **solo in
  valutazione** per portare la nuvola predetta nel sistema della mesh.
- **Lifting** — l'intera catena maschere 2D → punti 3D → voto sui vertici → istanze 3D.
