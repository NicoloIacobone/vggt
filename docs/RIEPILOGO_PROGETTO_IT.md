# Riepilogo del progetto (in italiano)

Sintesi corrente, aggiornata al 2026-08-06. **I numeri autorevoli stanno in `docs/RESULTS.md`** —
questo file li riassume, non li duplica: se un numero cambia lì, va cambiato anche qui.
La narrativa completa della testa precedente (ora ritirata) è archiviata in
`docs/old/RIEPILOGO_PROGETTO_IT_d4rt.md` e non fa più parte della storia corrente.

## 1. Obiettivo

Attaccare e allenare un decoder per **segmentazione di istanze 3D multi-view consistente** sopra
il backbone **VGGT-1B congelato**. VGGT non viene mai modificato: è il vincolo centrale del
progetto e la ragione per cui i risultati sono interessanti (i competitor pubblicati adattano il
backbone con LoRA). Supervisione: annotazioni ufficiali di istanze 2D di **ScanNet v2**.

Punto di aggancio: `aggregated_tokens_list[-1]` dell'aggregator — feature globali di scena
`F: [B, S, P, 2048]`. Il backbone gira sotto `no_grad` e le feature sono **cachate una volta per
scena**, motivo per cui un training dura minuti e non ore.

## 2. Il modello attivo: decoder MaskDINO

`models/maskdino/` — pixel decoder (token VGGT → piramide ViTDet a 3 livelli → encoder
MSDeformAttn) + decoder MaskDINO (selezione two-stage, anchor DAB, denoising, deep supervision).
~20.5 M parametri allenabili. Dettagli architetturali: `docs/MASKDINO.md`.

Estensioni multi-view:

- `--feature_mode bundle`: l'aggregator gira una volta su tutto il bundle invece che per frame.
- `--multi_frame`: **un solo set di query condiviso da tutti gli S frame del bundle**, con
  attenzione cross-frame. Una query = un'istanza in tutte le viste, per costruzione, senza
  matching post-hoc.
- `--anchor_3d`: l'anchor box 2D del decoder diventa un anchor 3D (x, y, z, log r) letto dalla
  testa POINT congelata di VGGT e proiettato in ogni vista. È un'**ablation**, non un contributo:
  il meccanismo è di FAST3DIS.
- `--num_frames` / `--eval_num_frames`: larghezza del bundle in training e in validazione.

## 3. I quattro righelli (non sono intercambiabili)

Questo è l'errore più facile da fare nel progetto. Ogni numero appartiene a un protocollo.

| righello | cosa misura | dove |
|---|---|---|
| **per-frame** | ogni frame valutato separatamente | `RESULTS.md` §2 |
| **per-bundle (multi-view)** | una IoU per istanza sul volume di maschere concatenato | `RESULTS.md` §3, §6 |
| **3D ufficiale ScanNet** | istanze 3D sulla nuvola di punti del benchmark, evaluator ufficiale | `RESULTS.md` §5 |
| **COCO** | verifica di correttezza della porta, non un risultato del progetto | `docs/MASKDINO_COCO.md` |

## 4. Risultati principali

**2D, split ufficiale 1201/312** (il righello onesto, `RESULTS.md` §6):

| | per-frame mIoU / AP50 | per-bundle mIoU / AP50 |
|---|---|---|
| single-frame | 0.624 / 0.662 | — |
| multi-frame (S=8) | 0.623 / 0.650 | 0.529 / 0.525 |
| multi-frame **S=16** | 0.627 / **0.662** | **0.549 / 0.552** |

**2D, split di progetto (val 0080–0089)**: MaskDINO 0.694 / 0.729 per-frame contro lo 0.451 /
0.294 della testa ritirata, e 0.539 / 0.515 per-bundle contro 0.367 / 0.199. Vedi `RESULTS.md`
§2–§3.

**3D, benchmark ufficiale** (`RESULTS.md` §5) — l'unico protocollo confrontabile con la
letteratura, e va letto sapendo che i numeri pubblicati sono **due protocolli diversi**:

| | AP / AP50 / AP25 | protocollo | classi |
|---|---|---|---|
| noi, default | 0.023 / 0.067 / 0.268 | unposed (geometria predetta da VGGT) | class-aware (18) |
| **noi, scoring class-agnostic** (manopole tarate) | **0.017 / 0.060 / 0.334** | unposed | **class-agnostic — l'unica riga confrontabile con le due qui sotto** |
| noi, `--anchor_3d` | **0.038 / 0.112 / 0.360** | unposed | class-aware (18) — versione class-agnostic in corso |
| FAST3DIS (pubblicato, backbone LoRA), 50 viste | 0.038 / 0.096 / 0.316 | unposed | **class-agnostic** |
| IGGT, **ri-valutato da FAST3DIS** (50 viste) | 0.028 / 0.112 / 0.287 | unposed | **class-agnostic** |
| noi, posed (`--transfer_mode gt_projection`) | 0.060 / 0.156 / 0.408 | posed (pose + depth GT) | class-aware (18) |
| SegVGGT (pubblicato, backbone LoRA) | 0.504 / 0.717 / 0.870 | **posed — non confrontabile con le righe unposed** | class-aware (18) |

Due precisazioni di provenienza, verificate sui paper il 2026-08-06:

- **`mAP` e `AP` sono la stessa metrica** (media su IoU 0.50:0.05:0.95; AP50/AP25 a soglia fissa).
  L'intestazione diversa fra la tabella di SegVGGT e quella di FAST3DIS non significa niente. A
  cambiare è il **setting**: FAST3DIS e IGGT ignorano le etichette semantiche (§4.4 del loro
  paper), noi e SegVGGT no. **L'abbiamo misurato invece di ragionarci** (job 9861563/9861564):
  scoring class-agnostic il nostro checkpoint fa **0.017 / 0.060 / 0.334** (default 0.013 / 0.050
  / 0.320). Confronto a parità di setting: **davanti su AP25** (0.334 contro 0.316 di FAST3DIS e
  0.287 di IGGT), **dietro di ~1.6–2.2× su AP50 e AP**. Ignorare le classi ci *abbassa* AP e AP50
  perché sostituisce la media su 18 classi — che da noi è retta da classi rare e distintive
  (toilet 0.508 AP50, che pesa 1/18) — con un unico ranking dominato dalle classi numerose e
  deboli (sedie 0.053) e da `otherfurniture`, che la nostra testa a 19 classi non predice (0.000).
- **Il numero di IGGT non viene dal paper di IGGT**, che su ScanNet non riporta nessun AP (solo
  tracking, ricostruzione e semantica open-vocab, su 10 scene × 8–10 immagini). È la
  ri-valutazione fatta da FAST3DIS.

**Perché i numeri "alti" che si ricordano sono un'altra famiglia.** Nella stessa Table 1 di
SegVGGT: Mask3D 55.2 / 73.7 / 85.3, Relation3D 62.5 / 80.2 / 87.0, SegDINO3D 64.0 / 81.5 / 88.9 —
tutti con **nuvola di punti o RGB-D in input**. L'unico baseline solo-immagini di quella tabella,
OneFormer3D†, fa **5.4 / 10.2 / 17.4**, sotto di noi pur essendo nel protocollo posed.

## 5. Le conclusioni che contano

1. **Il collo di bottiglia era l'architettura, non i dati.** La testa precedente peggiorava con
   più dati; MaskDINO guadagna +0.26 AP50 passando da 50 a 490 scene. Il modello è tuttora
   *data-limited*.
2. **La scala dei dati domina ogni singolo componente**: ≤0.05 AP50 per qualunque ingrediente
   MaskDINO rimosso, contro +0.26 dalla scala.
3. **L'attenzione cross-frame serve all'identità, non al riconoscimento.** Toglierla lascia
   invariato il numero di istanze trovate ma fa saltare `bundle_id_switch` da 0.498 a 0.682.
4. **Il risoluzione delle maschere non è il vincolo**: la griglia 37×37 ha un tetto GT-only di
   0.956 AP50 contro lo ~0.69 del modello. Vincola il riconoscimento.
5. **Sul righello 3D il collo di bottiglia è il lifting, non il decoder.** L'errore di
   registrazione mediano (0.14 m) è dell'ordine del raggio di voto, e solo ~15 % dei vertici
   riceve un voto. Il ponte 2D→3D costa un fattore 2.3 di AP50.
6. **`bundle_AP50` a S=8 è un cattivo proxy del righello 3D**: `--anchor_3d` è piatto in 2D e
   vale +67 % AP50 in 3D. Identità e AP sono assi separati.
7. **Allargare il bundle da 8 a 16 viste aiuta**: +0.027 per-bundle AP50 e `bundle_id_switch` da
   0.498 a 0.385, a parità di frame budget.

## 6. Onestà metodologica (le regole che ci siamo dati)

- Un numero 3D è **riportabile** solo se il checkpoint non ha mai visto le scene di val-312.
  I checkpoint allenati su 0000–0489 producono numeri **diagnostici**.
- I due protocolli 3D si stampano come **due colonne**, mai fusi. SegVGGT non sta barando: il
  loro modello è unposed quanto il nostro, usano la geometria GT solo per trasferire maschere
  già finite ai fini dello scoring — e lo scrivono nel paper (*"we utilize the ground-truth depth
  maps and camera poses during this mapping stage for fair comparison"*).
- Ogni run 3D produce ora **anche** il numero class-agnostic (`results_class_agnostic`), così il
  confronto con FAST3DIS/IGGT può essere fatto a parità di setting invece che a parole.
- Il protocollo posed è **licenziato da un oracolo**: rendendo la GT 3D attraverso la stessa
  proiezione si torna al 99.99 % sull'istanza corretta. Senza quell'oracolo il numero non si cita.
- Le manopole di lifting sono state ri-sweepate sul checkpoint pulito; l'argmax dello sweep **non**
  è il titolo, perché lo sweep gira su val.

## 7. Dove guardare

- `docs/MASKDINO.md` — architettura e protocolli (documento primario).
- `docs/RESULTS.md` — tutti i numeri.
- `docs/COMMANDS.md` — tutti i comandi eseguibili.
- `docs/RELATED_WORK.md` — posizionamento rispetto ai competitor.
- `docs/todo.md` — lavoro aperto.
- `docs/old/` — archivio (inclusa la testa ritirata e la sua narrativa completa).
