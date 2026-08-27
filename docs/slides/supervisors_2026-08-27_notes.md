# Note del relatore — deck `supervisors_2026-08-27.md`

Documento di preparazione, **non** una slide. Per ogni slide: a cosa serve, come aprirla (frase
pronta, in inglese, da dire ad alta voce), cosa dire, cosa **non** dire, e le domande che quella
slide invita.

Regole che valgono per tutto il talk:

- **Ogni numero 3D va accompagnato da due etichette**: *posed / unposed* e *class-aware /
  class-agnostic*. Un numero senza etichette non è confrontabile con nulla — la slide 3 esiste apposta.
- **I numeri 2D interni non vanno mai accostati a un numero pubblicato** di un competitor. Sono
  nostro codice di metrica su maschere per-vista: servono a scegliere checkpoint e a ordinare
  ablation, non a fare classifica.
- **Le righe con dati extra restano recintate** dalla riga headline (che è ScanNet-only).
- **Tutto ciò che è addestrato su RE10K è supervisionato da SAM2** — le maschere sono output di un
  modello, non ground truth. Va detto ogni volta.
- Fonte unica di ogni numero: `docs/FACTSHEET.md`. Se un numero non è lì, non è autorizzato.

---

## Slide 1 — Titolo

**Punto della slide:** dichiarare subito che la riunione ha due decisioni da prendere, non è un
aggiornamento di stato.

**Apertura:** *"Two things I need from you today: which of the last ablations to run, and whether
this becomes a CVPR submission in November. Everything else on these slides is evidence for those
two."*

**Cosa dire:** il progetto ha una headline che regge il confronto con la letteratura; le due
domande aperte sono (a) come chiudere il buco nella tabella delle ablation e (b) se scrivere il
paper. Anticipa che la struttura del talk è: setup → come si leggono i numeri → metodo → righello →
risultati → decisioni.

---

## Slide 2 — The setup

**Punto della slide:** far capire in un minuto *che tipo* di sistema è. Le tre cose che contano:
backbone congelato, feature in cache, supervisione solo 2D.

**Apertura:** *"Three facts define this system: the backbone is frozen, its features are computed
once and cached, and nothing but 2D masks ever supervises it."*

**Cosa dire:**

- *Frozen*: VGGT-1B non viene mai aggiornato. Nessun finetuning, nessun LoRA. Tutti i competitor
  adattano il backbone; noi no, ed è una scelta che va rivendicata (slide 18), non giustificata.
- *Cached features* — è la domanda che ti faranno: **il backbone congelato viene eseguito una volta
  per scena, prima dell'addestramento, e i token che produce vengono salvati su disco. Il training
  loop legge solo quei token; il backbone non gira mai dentro il ciclo di training.** Conseguenza
  diretta: si addestra solo la testa e il costo scende a ~0.8 GPU-days contro i ~16 del competitor
  più vicino. È anche il motivo per cui una run dura minuti/ore e non giorni.
- *Supervision*: **solo maschere 2D, mai un'etichetta 3D.** Tre livelli, da tenere distinti:
  la run headline usa **solo** le annotazioni ufficiali di istanza 2D di ScanNet v2; le run
  "extra data" aggiungono ScanNet++ e Infinigen (annotazioni di istanza per-frame che arrivano già
  pronte con InsScene-15K) e sono addestrate **class-agnostic**; le run con RE10K sono
  **supervisionate da SAM2** e vanno etichettate ogni volta.

**Cosa NON dire:** non dire "usiamo ScanNet" e basta se poi mostri le righe extra-data — è
esattamente l'ambiguità che questa slide rimuove. E non presentare le righe RE10K come "più dati"
senza dire che le maschere sono output di un modello.

**Domanda attesa — "Quindi il modello non vede mai geometria 3D?"** In addestramento no, mai. La
geometria 3D compare solo nel *ponte* di valutazione (slide 7) e, nella variante con ancore 3D,
come *prior posizionale* letto dalla testa di punti di VGGT — che è comunque predetta dal modello,
non ground truth.

---

## Slide 3 — How to read every number in this deck

**Punto della slide:** è la slide-glossario. Serve a impedire che qualcuno confronti due numeri
nostri prodotti con protocolli diversi, o un nostro numero con un numero pubblicato sotto un altro
setting. Se salti questa slide, il resto del talk diventa fragile.

**Apertura:** *"Before any number: in this literature the same model can score four different
things depending on the setting. So every number here carries its labels."*

**Cosa dire:**

- **La tripla `AP / AP50 / AP25`** è sempre in quest'ordine. `mAP` ≡ `AP`: se una tabella
  pubblicata scrive `mAP` e un'altra `AP` non vuol dire nulla, è la *stessa* metrica. Ciò che
  cambia è il setting.
- **Unposed vs posed** è il ponte 2D→3D. *Unposed*: profondità e camere predette da noi, quindi il
  numero misura maschere **per** geometria. *Posed*: pose, intrinseci e depth del sensore di
  ScanNet, quindi misura **solo** le maschere. Vale un fattore **2.3× in AP50** in modo costante —
  è la variabile di protocollo più grossa dell'intero campo.
- **Class-agnostic vs class-aware**: etichette ignorate, o media sulle 18 classi. FAST3DIS e IGGT
  pubblicano **solo** class-agnostic, SegVGGT è class-aware. Noi calcoliamo entrambe le colonne per
  ogni run — che è ciò che rende possibili entrambi i confronti.
- **Views per scene**: quante viste della stessa scena entrano in un singolo forward pass.
- **`id_switch` / `view_consistency`**: i due numeri di *identità* cross-vista, e **non** sono ciò
  che misura l'AP. Presa una query accoppiata a un'istanza GT sull'intero bundle:
  `view_consistency` è la frazione di viste in cui **quella** query segmenta ancora l'istanza
  (**più alto è meglio**); `id_switch` è la frazione di viste in cui un'**altra** query la spiega
  meglio, cioè in cui l'identità "salta" da una query all'altra (**più basso è meglio**). L'AP si
  calcola sull'istanza 3D **già fusa**: vede l'identità solo di rimbalzo — un'identità rotta le
  arriva come maschera peggiore, mai come numero a sé. Per questo un meccanismo di identità si
  giudica su `id_switch` **e** sul righello 3D, mai sull'AP 2D da sola.

**Perché è importante dirlo qui:** i due cluster di numeri pubblicati (FAST3DIS/IGGT intorno a
0.03–0.04 AP, SegVGGT a 0.50) sembrano incoerenti finché non si sa che sono due ponti diversi.
Se lo spieghi ora, la slide 12 non richiede difese.

---

## Slide 4 — The method

**Punto della slide:** una sola idea — *una query è un'istanza in tutte le viste, per costruzione*.
E il tensore di output che la rende concreta.

**Apertura:** *"The whole model output is one tensor, and the reason it is one tensor is the point
of the method."*

**Cosa dire:**

- Il decoder emette `pred_masks [B, N, S, h, w]`. Leggi la tabella ad alta voce: **B** bundle nel
  batch (un bundle = le S viste di una scena), **N** query condivise da tutto il bundle (in fase di
  scoring ne teniamo le prime 100), **S** viste nello *stesso* forward pass (~17 per scena ScanNet),
  **h × w = 37 × 37** che è la griglia di patch di VGGT, cioè la risoluzione su cui le maschere
  vengono predette.
- La conseguenza: la query numero *n* è **lo stesso oggetto in tutte le S viste**. Non c'è un passo
  di matching, non c'è fusione a posteriori, non c'è tracking. La consistenza multi-vista è una
  proprietà della rappresentazione, non un post-processing.
- Il confronto retorico utile: chi fa fusione a posteriori deve decidere *dopo* che la maschera
  della vista 3 e quella della vista 7 sono lo stesso oggetto. Noi non abbiamo quel passo, quindi
  non abbiamo neanche i suoi errori — ma abbiamo il costo opposto (slide 14: le bundle features
  costano −0.048 di AP per-frame).

**Se chiedono perché 37×37 è così poco:** rimanda alla slide 15, punto 3 — è **misurato** che la
risoluzione non è il collo di bottiglia: su quella griglia il ceiling con GT è 0.956 AP50 e il
modello sta a ~0.69. Quello che lega è il riconoscimento.

---

## Slide 5 — Two ways to anchor a query

**Punto della slide:** rispondere alla domanda "ma il decoder ha una sola modalità?". No: ne ha
**due**, ed è l'ablation principale dello studio.

**Apertura:** *"There are two versions of this decoder, and the difference between them is the one
experiment nobody in the field has run."*

**Cosa dire:**

- **Versione base — box 2D.** Ogni query porta una box 4-d **per vista**, raffinata layer dopo
  layer: è la ricetta DAB/DINO standard. Il prior posizionale è bidimensionale e vive dentro la
  singola immagine.
- **Versione con ancore 3D (`--anchor_3d`).** Ogni query porta invece **un punto 3D più un raggio**,
  uno **per scena** e non per vista, letto dalla testa di punti congelata di VGGT. Per sapere dove
  guardare nella vista *f*, l'ancora viene proiettata con un *soft nearest-patch* (un softmax sulle
  distanze tra l'ancora e le posizioni 3D dei patch di quella vista), **senza usare intrinseci né
  estrinseci**: non è una proiezione prospettica, è una media pesata. Questo è ciò che rende la
  versione 3D utilizzabile anche quando le camere predette sono imprecise. *(La slide aveva una
  terza colonna "serve la geometria della camera?": era "no" in entrambe le righe, quindi non
  distingueva nulla ed è stata tolta — il fatto è ora un bullet unico sotto la tabella, valido per
  entrambe le varianti.)*
- Vincolo da conoscere: le ancore 3D hanno senso **solo** se la query è già condivisa tra le viste
  (altrimenti un'ancora "per scena" non vuol dire niente) e richiedono le feature calcolate in
  modalità bundle.
- **La riga headline è quella con le ancore 3D**: valgono **+66 % di AP50 3D in entrambi i ponti** e
  riducono gli scambi di identità tra viste (−0.089), pur essendo neutre in 2D. È l'esempio pulito
  del punto 4 della slide 15: *un meccanismo può comprare identità senza comprare accuratezza*.
- Chiudi con l'onestà: **nessuno dei due meccanismi è nostro.** Le query condivise tra viste sono di
  SegVGGT, le ancore 3D sono di FAST3DIS. Quello che non ha fatto nessuno è **metterle una contro
  l'altra dentro lo stesso decoder, con lo stesso backbone, gli stessi dati e lo stesso protocollo**.

**Cosa NON dire:** mai "abbiamo inventato le ancore 3D". È esattamente il claim che la slide 17
smonta, e un revisore lo smonterebbe in dieci secondi.

---

## Slide 6 — The ruler: the evaluator

**Punto della slide:** stabilire che il righello non è nostro. È lo script ufficiale, verificato, e
lo stesso su tutti e quattro i benchmark.

**Apertura:** *"The evaluator is not ours, and that is deliberate."*

**Cosa dire:**

- È il **benchmark ufficiale di instance segmentation 3D di ScanNet**, e lo script di valutazione è
  quello ufficiale **portato dentro il repo** (vendored): stesse soglie di overlap, stesso matching
  ordinato per confidenza, stessa gestione dei *void*. Non abbiamo scritto la metrica.
- **Il controllo di licenza**: su ciascuno dei quattro dataset, il GT di quel dataset dato in pasto
  all'evaluator *come se fosse una predizione* deve fare esattamente **1.000 / 1.000 / 1.000**. Lo
  fa su tutti e quattro. È il test che dimostra che l'adattatore del dataset e il ponte non stanno
  perdendo per strada delle istanze.
- **Lo stesso evaluator misura tutti e quattro i benchmark** (ScanNetv2, ScanNet200, ScanNet++,
  Replica). Cambia solo l'adattatore del dataset: le sue nuvole di punti, il suo GT di istanze, la
  sua tassonomia. Testa, ponte e lifting non si toccano.
- Sugli altri tre riportiamo **solo class-agnostic**: la nostra testa ha 19 classi ScanNet e quelle
  tassonomie non sono le nostre. Inventare una corrispondenza sarebbe fabbricare un confronto, non
  misurarlo. Su ScanNet200 le etichette esistono ma sono 200: stesse scene, stesse tar, tassonomia
  diversa — per questo "costa zero dati" (slide 13).

**Domanda attesa — "Quindi su ScanNet++ e Replica usate il benchmark di ScanNet?"** Usiamo il
**codice** dell'evaluator di ScanNet sul GT *di quei dataset*. È la stessa metrica applicata a
un'altra verità di riferimento, non il GT di ScanNet applicato altrove. E su quei tre **non
esistiamo in classifica**: nessuno pubblica una riga confrontabile, quindi quei numeri sono
evidenza interna, non un confronto.

---

## Slide 7 — The ruler: the two bridges

**Punto della slide:** spiegare *come* le maschere 2D diventano istanze 3D, e perché quel passo da
solo vale 2.3×.

**Apertura:** *"Our model predicts 2D masks. The benchmark scores 3D instances. Everything
interesting happens in between."*

**Cosa dire:**

- **Ponte unposed.** Ogni pixel di ogni maschera viene proiettato nello spazio usando la profondità
  e le camere **predette dal modello**. Problema: VGGT ricostruisce una scena solo **a meno di una
  rotazione, una traslazione e una scala globale** — non sa quanto è grande la stanza in metri.
  Quindi, per poter confrontare con la mesh del benchmark, la predizione finita viene **portata nel
  sistema di riferimento del benchmark** da una **trasformazione di similarità**:
  **`Sim(3)` = rotazione + traslazione + una scala globale** (in contrapposizione a `SE(3)`, che
  sarebbe solo rototraslazione). La stima è in forma chiusa dai **centri delle camere** predetti
  contro quelli veri, poi viene raffinata con **ICP** (*iterative closest point*: si allinea
  iterativamente la nuvola predetta ai vertici della mesh, riaggiornando la scala a ogni giro).
  **Frase da dire testualmente: le pose ground truth servono solo a posare la predizione già
  finita, non entrano mai nell'inferenza.** È esattamente la convenzione di FAST3DIS → asse
  *matched*.
- **Ponte posed.** Pose, intrinseci e depth del sensore veri: il ponte è esatto per costruzione, e
  quindi il numero misura **solo la qualità delle maschere 2D**. È il *"geometric GT"* di SegVGGT,
  riprodotto — e **certificato**, non assunto: l'oracolo (GT dentro il ponte) restituisce il
  **99.99 %** dei vertici annotati assegnati alla propria istanza.
- **Il ponte da solo vale 2.3× di AP50.** Ripetilo: è il motivo per cui i numeri pubblicati sembrano
  incoerenti tra loro e il motivo per cui noi pubblichiamo **entrambe** le colonne.
- **Ceiling del setup:** le maschere GT spinte attraverso il ponte posed fanno
  **0.828 / 0.948 / 0.974**. Non è 1.0 perché ~17 viste non coprono tutta la scena: è il tetto del
  budget di viste, non un difetto del modello.
- **Non far collegare due numeri che non c'entrano** (errore già capitato leggendo questa slide):
  il **2.3×** è il rapporto *posed ÷ unposed* sullo **stesso** checkpoint e sulle **stesse**
  maschere. Lo **0.828 / 0.948 / 0.974** è tutt'altro: è l'**oracolo**, cioè le maschere **GT**
  dentro il ponte posed. Non è "2.3× di qualcosa" e non si confronta con la headline unposed
  (0.042 / 0.138 / 0.504): il suo termine di paragone è la colonna **posed**, dove il nostro meglio
  è 0.088 / 0.260 / 0.572.

**Domanda attesa — "Usare le pose GT per l'allineamento non è barare?"** No, e la distinzione è
netta: nel ponte unposed le pose GT entrano **dopo** che la predizione è completa, solo per metterla
nello stesso sistema di coordinate della mesh — senza, il confronto sarebbe indefinito perché la
scala è arbitraria. È la stessa cosa che fa FAST3DIS. Il ponte *posed*, invece, usa la geometria
vera **dentro** il trasferimento, ed è per questo che lo riportiamo come colonna separata e
etichettata.

---

## Slide 8 — The headline

**Punto della slide:** è **la** slide del progetto. Un solo claim.

**Apertura:** *"This is the row the whole project is for. Same benchmark, same evaluator, same
bridge, same label setting as the two published feed-forward competitors."*

**Cosa dire:**

- Setting: benchmark 3D ufficiale di ScanNet, **unposed** e **class-agnostic** — cioè esattamente il
  setting in cui i due competitor pubblicano.
- Il claim, parola per parola: **su due seed, guidiamo su AP50 (1.34–1.44×) e su AP25 (1.53–1.59×),
  siamo in PARITÀ con FAST3DIS su AP, e siamo avanti a IGGT su tutte e tre** — con backbone
  congelato, circa un terzo delle loro viste, e tutti i parametri di lifting ai valori di default.
- La riga extra-data (0.057 / 0.166 / 0.516) resta **recintata**: è la norma del campo (lo fa anche
  il README di MaskDINO). Il claim sul *meccanismo* poggia sulla riga ScanNet-only; il claim sullo
  *scaling* su quella extra-data.

**Cosa NON dire — errore già commesso una volta in questo progetto:** *"avanti su tutte e tre"* per
la riga ScanNet-only. **Su AP è un pareggio** (0.039–0.042 contro 0.038, dentro la nostra dispersione
di seed). "Avanti su tutte e tre" è autorizzato **solo** per la riga extra-data, e solo con
l'etichetta extra-data attaccata.

**Attribuzione obbligatoria:** IGGT va sempre citato come *"as re-evaluated by FAST3DIS"*. IGGT non
pubblica **nessun** AP su ScanNet: quella tripla è la ri-valutazione fatta da FAST3DIS.

---

## Slide 9 — The four caveats

**Punto della slide:** dire tu i limiti prima che li dicano loro. Rende credibile tutto il resto.

**Apertura:** *"Four caveats travel with that table. I would rather state them than be asked."*

**Cosa dire:** i quattro punti sono già sulla slide; il modo di raccontarli conta.

1. **Due seed, ma una run per cella** — e soprattutto **una sola riga pubblicata per competitor**:
   la debolezza statistica sta dalla *loro* parte, non solo dalla nostra.
2. **Non siamo training-matched, e l'asimmetria è a nostro favore.** Entrambi i competitor sono
   *zero-shot* su ScanNet (FAST3DIS addestra solo su Aria/ASE, IGGT su InsScene-15K), noi ci
   addestriamo. Va detto ad alta voce, e va detto che le due run "no-ScanNet" che chiudono questo
   asse sono già in coda (slide 19).
3. **Il segno del class collapse dipende dal checkpoint.** La stessa ricetta *senza* ancore 3D fa
   0.017 / 0.060 / 0.334 class-agnostic **con i parametri di lifting ottimizzati** (0.013 / 0.050 /
   0.320 ai default): avanti su AP25, ~2× indietro su AP50/AP. Quindi non è un confronto
   default-contro-default e l'etichetta va portata.
4. **Due asimmetrie corrono nell'altro senso**: il nostro backbone è congelato, e usiamo ~17 viste
   contro 50.

**Nota per te:** i punti 2 e 4 insieme sono il messaggio vero — le asimmetrie non sono tutte dalla
stessa parte, e sono tutte dichiarate. È questo che rende la riga headline difendibile.

---

## Slide 10 — The matched axes

**Punto della slide:** dimostrare che il confronto è stato costruito **con il setting del
competitor**, asse per asse, e non con il nostro.

**Apertura:** *"Every competitor-facing row is produced under the competitor's own setting. Here is
the audit, axis by axis."*

**Cosa dire:** non leggere tutte le righe. Leggine tre e dichiara il resto:

- evaluator ufficiale vendored → *matched*;
- il ponte unposed è la stessa convenzione Sim(3)+ICP di FAST3DIS → *matched*;
- il ponte posed è il "geometric GT" di SegVGGT, **certificato al 99.99 %** → *matched, e verificato,
  non assunto*;
- e poi la riga che ti fa risparmiare una discussione: **kept queries**. SegVGGT ne tiene 600, noi
  100 — e l'abbiamo **misurato**: 0.138 → 0.140. È dentro il rumore, quindi **quella spiegazione del
  gap è cancellata**. Averlo misurato e averla cancellata è più forte che averla lasciata come
  scusa.

---

## Slide 11 — The three gaps

**Punto della slide:** i tre assi **non** matched, ciascuno con la direzione dichiarata. È la slide
dell'onestà, e va detta con tono neutro.

**Apertura:** *"Three axes are not matched. Two of them are being closed, one never will be — and
they do not all run in the same direction."*

**Cosa dire:**

- **Viste (~17 contro 50, o 75–100 di SegVGGT): corre CONTRO di noi.** È il meglio disponibile oggi
  perché il sottoinsieme di frame che abbiamo estratto è quello; l'export denso è in coda (slide 19).
- **Dati di addestramento: corre A FAVORE nostro** — loro sono zero-shot su ScanNet, noi no. Le due
  run "no-ScanNet" chiudono questo asse.
- **Compute: ~0.8 contro ~16 GPU-days, e non è colmabile.** Va presentato come **forza, non come
  scusa**: è la conseguenza diretta del backbone congelato con feature in cache.
- **ASE** (il dataset di FAST3DIS): 9.2 TB e con il 40 % della lista di scene non pubblicata.
  Fuori portata in modo permanente — e non è un nostro limite, è una proprietà della loro release.

**La frase che chiude la slide:** *"a comparison with no state named is not ready to be shown"* —
ogni riga verso un competitor è **matched**, **closest-available-and-declared**, oppure
**permanentemente impossibile**. Tre stati, mai mescolati.

---

## Slide 12 — The posed comparison vs SegVGGT, e dove va il gap

**Punto della slide:** mostrare le stesse maschere sotto entrambi i ponti, e far vedere che il
salto è sistematico (2.3× su ogni riga).

**Apertura:** *"Same masks, two bridges. The ratio is the same on every row, which is what makes it
a property of the protocol rather than of a checkpoint."*

**Cosa dire:**

- Le righe sono quattro checkpoint nostri: la **run di controllo** (quella su cui è ancorata la
  decomposizione della seconda metà della slide), quella con **ancore 3D**, quella con **bundle più largo
  (16 viste)**, e le due cose insieme.
- Il fatto notevole non è nessuna riga singola: è che **il rapporto posed/unposed è 2.2–2.3× su
  tutte**. Cioè il ponte è una costante moltiplicativa del protocollo, non un effetto di un
  particolare modello.
- L'**oracolo** (GT attraverso il ponte posed, 0.828 / 0.948 / 0.974) dice dove sta il tetto con ~17
  viste.
- SegVGGT, pubblicato e posed: 0.504 / 0.717 / 0.870. **Siamo dietro, e va detto in chiaro** — poi
  la seconda metà della slide dice di quanto e perché.

**Regola:** le due colonne viaggiano sempre insieme. Non mostrare mai la colonna posed da sola:
fuori contesto sembra un risultato migliore di quello che è.

**Seconda metà della slide — punto:** trasformare "siamo 10 volte dietro" in "sappiamo esattamente dove vanno quelle
10 volte, e due terzi delle cause sono scelte nostre".

**Aggancio alla seconda metà:** *"We are behind SegVGGT by about ten times. That number decomposes, and the
decomposition is the interesting part."*

**Cosa dire:**

- Dei **×10.7**: **2.3 è il ponte** (protocollo, non modello) e **~4.6 è reale**.
- Il ~4.6 è comprato con quattro cose che noi **abbiamo scelto di non avere**: backbone adattato con
  LoRA contro congelato; 75–100 viste contro ~17; maschere 259×196 contro 37×37; 600 query tenute
  contro 100 — e **quest'ultima l'abbiamo misurata ed è neutra** (0.138 → 0.140), quindi esce
  dall'elenco delle spiegazioni.
- **Attenzione, e va detto:** la decomposizione è ancorata **alla riga di controllo**
  (0.067 unposed → 0.156 posed contro 0.717 di SegVGGT). Le righe migliori della tabella qui sopra
  hanno un
  residuo *più piccolo*, ma **non abbiamo un numero autorizzato per quel residuo**. Quindi si cita
  la decomposizione **nominando la riga**, e non la si ricalcola mai su un'altra riga.
- **Se qualcuno chiede "ma perché il vostro AP è così basso in assoluto?"**: la baseline
  image-only nella **tabella di SegVGGT stessa**, OneFormer3D†, fa **5.4 / 10.2 / 17.4**. Quella
  tabella riporta l'AP in percentuale, quindi sulla scala di queste slide quella baseline sta nello
  stesso intervallo delle nostre righe posed. Serve a ricalibrare l'aspettativa: in questo
  benchmark, in questo setting, i numeri sono piccoli per tutti tranne SegVGGT.

**Nota sulla fusione:** questa slide era due (il confronto posed e la decomposizione del gap). Sono state unite perché la seconda senza la prima non si legge: la tabella fornisce la riga di controllo su cui la decomposizione è ancorata.

---

## Slide 13 — The other three benchmarks

**Punto della slide:** due cose che ScanNet da solo non può mostrare — dove pagano i dati extra, e
dove esattamente fallisce lo zero-shot.

**Apertura:** *"ScanNet alone cannot show two things: what the extra data actually buys, and where
the system breaks out of domain."*

**Cosa dire:**

- **Non è un'ablation**: non si toglie niente al modello. È **un solo checkpoint** misurato su
  quattro dataset, quindi misura *transfer* — quanto lontano arriva lo stesso modello. E i tre
  benchmark non sono tutti "fuori dominio": **ScanNet200 sono le stesse scene** con una tassonomia
  a 200 classi (dominio identico, compito più difficile), mentre **ScanNet++ e Replica sono
  davvero out-of-domain**.
- Prima riga: il checkpoint **ScanNet-only**, cioè lo stesso della headline, sui tre altri benchmark.
- Seconda riga: **con i dati extra**. Il confronto tra le due è il punto della slide: su ScanNet++
  (0.009 → 0.019) e su Replica (0.006 → 0.040) i dati extra pagano **molto più** che su ScanNet.
  **È l'unica misura nel progetto di qualcosa che il righello ScanNet non vede.**
- Il risultato negativo, che è forse il più informativo del progetto: **lo zero-shot muore sotto il
  ponte unposed** — ogni cella fuori dominio è 0.000 AP / 0.000–0.001 AP50 su tutte e quattro le run
  di dati — **e sopravvive, debolmente, sotto quello posed**. Poiché l'unica differenza tra i due
  ponti è la geometria, **il fallimento è localizzato nella geometria, non nelle maschere**. Le
  maschere fuori dominio ci sono; è la ricostruzione feed-forward che non regge.
- **ScanNet200 costa zero dati aggiuntivi**: stesse scene, stesse tar, tassonomia diversa.
- Chiudi con la recinzione: **qui non esiste una riga pubblicata confrontabile**. È evidenza, non
  classifica.

---

## Slide 14 — What actually buys the result

**Punto della slide:** ordinare le leve per importanza misurata, e far vedere che l'ultima colonna
("measured on") nasconde il problema che diventerà la decisione (a).

**Apertura:** *"Ranked by what we measured, not by what we expected. And read the last column —
it is the reason for the first decision I am asking you for."*

**Cosa dire:**

- **Il primo posto sono i dati** (+0.26 AP50 da 50 a 490 scene): domina tutto il resto, e nessuna
  scelta architetturale si avvicina.
- **"Perché 50→490 se ormai siamo a 3520 scene?"** — domanda attesa, e la risposta è "righelli
  diversi". Quella riga è la *curva di scala* del righello interno su project-val, che esiste solo
  fino a 490 scene: è la riga storica che ha stabilito la direzione. Le run grandi vivono sullo
  split ufficiale 1201/312 e stanno sotto la tabella: per-bundle AP50 **0.548** (ScanNet, 1201
  scene) → **0.604** (+ScanNet++ +Infinigen, 3520 scene, a convergenza), `id_switch` 0.441 → 0.414;
  il loro corrispettivo 3D è la riga extra-data della headline (0.042 / 0.138 / 0.504 →
  0.057 / 0.166 / 0.516). Le due curve non si sommano e non si confrontano. **Caveat da dire da
  soli:** la riga a 3520 scene è confrontata *a convergenza*, non a passi uguali — "più dati" e
  "più compute" non sono ancora separati, ed è ciò che chiude la seconda riga della slide 19.
- Poi le due leve multi-vista (+0.183 e +0.147), che però portano il ⚠ perché sono misurate **solo
  in 2D**.
- Poi le leve che hanno un numero 3D: **ancore 3D** (+66 % di AP50 3D in entrambi i ponti),
  **larghezza del bundle** 8→16 viste (+46 % AP50 3D unposed) e i **parametri di lifting**.
- In fondo, i risultati **negativi**, che sono altrettanto importanti: **nessun singolo componente
  del decoder** (two-stage, encoder, denoising, box init) vale più di 0.046, e la **risoluzione
  delle maschere** è neutra. Cioè: non c'è un trucco architetturale che stiamo trascurando.
- L'ultima colonna dice su quale righello ogni Δ è stato misurato. **Due righe dicono "solo 2D".**
  Tieni il dito lì e passa alla slide 16.

---

## Slide 15 — Four conclusions

**Punto della slide:** le quattro conclusioni che sopravvivono a tutto il resto. Se il pubblico
ricorda solo una slide di contenuto tecnico, deve essere questa.

**Apertura:** *"Four things I would defend in a rebuttal."*

**Cosa dire:**

1. **Limitati dai dati, non dall'architettura.** La prova non è la curva: è che il checkpoint
   *leak-free* addestrato su 1201 scene **batte** quello che aveva **visto** le scene di validazione
   (0.083 contro 0.052 AP50, righello 3D). Più dati battono la fuga di dati — un risultato
   controintuitivo che chiude la questione.
   **Cos'è il "leak", se lo chiedono** — ed è una domanda giusta, perché oggi non esiste più: prima
   che il progetto passasse allo split ufficiale, alcuni checkpoint erano addestrati sulle scene
   0000–0489, che **si sovrappongono** alle scene su cui venivano poi validati. Il modello veniva
   cioè valutato in parte su scene già viste, e il numero era gonfiato. Da quando si usa lo split
   ufficiale **1201 train / 312 val**, disgiunti, il problema non c'è più: **ogni numero di questo
   deck è leak-free**. L'unico punto in cui la parola ritorna è la slide 16: il checkpoint
   `feature_mode single` che esiste già è uno di quelli vecchi, ed è esattamente per questo che il
   suo numero sarebbe solo diagnostico e serve una run nuova.
2. **A legare ora è il lifting, non il decoder** — AP25 è circa **4×** AP50: cioè gli oggetti li
   troviamo (AP25), ma i confini 3D non sono abbastanza precisi da superare la soglia più severa.
   Il problema è il ponte, non le maschere.
3. **E non è nemmeno la risoluzione**: sulla griglia 37×37 il GT fa 0.956 AP50, il modello ~0.69.
   Il collo di bottiglia è il **riconoscimento**.
4. **Riconoscimento e identità cross-vista sono assi separati.** Un meccanismo può comprare identità
   senza comprare accuratezza — le ancore 3D sono esattamente questo caso. Corollario operativo:
   un meccanismo di identità si valuta sul righello 3D, mai sull'AP 2D da sola.

Ripeti prima di iniziare: **ogni Δ va letto contro la dispersione di seed misurata, 0.009**. Sotto
quella soglia è rumore.

---

## Slide 16 — The hole in the ablation table → decisione (a)

**Punto della slide:** è la prima domanda concreta. Non è "abbiamo un problema", è "ecco due prezzi,
scegliete".

**Apertura:** *"Here is the first decision. The two strongest levers in this study have no 3D
number, and putting them on the 3D ruler costs two very different amounts."*

**Cosa dire:**

- Il problema in una frase: la headline vive sul righello 3D, ma le **due leve più decisive** —
  cross-frame attention (+0.183) e bundle features (+0.147), cioè **20× e 16×** la dispersione di
  seed — sono misurate **solo in 2D**. Tutte le altre leve (ancore 3D, larghezza bundle, lifting)
  hanno un numero 3D. Queste due no.
- Perché conta per il paper: un revisore chiederà perché la tabella delle ablation non è sullo stesso
  righello della tabella dei risultati. È una domanda giusta.
- **I due prezzi, e non vanno mai mediati:**
  - *"niente cross-frame attention"*: il checkpoint **esiste già** (job 9503176), addestrato sullo
    split ufficiale 1201, senza leak → costa **una singola run di valutazione 3D**. Praticamente
    gratis.
  - *"feature per-frame"*: il checkpoint esistente (job 8950613) è addestrato su scene 0000–0489,
    che **si sovrappongono al validation split** → quel numero sarebbe *leaked* e quindi solo
    diagnostico. Serve **una nuova run di addestramento**.
- La domanda che poni: **facciamo solo la metà gratis, o anche la nuova run?**

---

## Slide 17 — Positioning → decisione (b)

**Punto della slide:** dire per primi ciò che un revisore direbbe contro di noi. È l'unico modo di
avere credibilità sulla slide successiva.

**Apertura:** *"Before I claim anything: 'frozen VGGT plus a decoder' is the dominant pattern of the
last twelve months. The architecture on its own is not a contribution — so let me be precise about
what is and is not ours."*

**Cosa dire:** la tabella si legge riga per riga come "meccanismo → di chi è già → cosa resta a noi".

- **Query condivise tra le viste → sono di SegVGGT.** Questa è la riga che va spiegata bene, perché
  scritta da sola è criptica. Significato: SegVGGT ha **già pubblicato** l'idea di far attraversare
  a un insieme di query tutte le viste di una scena su un backbone della famiglia VGGT — 400 query,
  dentro tutti e 24 i layer dell'aggregatore. Quindi **noi non possiamo rivendicare quel meccanismo
  come nuovo**: la nostra modalità multi-frame è la *stessa classe di idea*, realizzata **fuori** dal
  backbone congelato invece che dentro. Va presentata come **confronto controllato contro il nostro
  stesso modello single-frame**, non come meccanismo nuovo. (Differenza tecnica, se la chiedono:
  loro modificano il backbone con LoRA e mettono le query dentro l'aggregatore; noi agganciamo un
  solo hook all'ultimo layer e non tocchiamo un solo parametro del backbone.)
- **Ancore 3D → sono di FAST3DIS.** Quindi le nostre sono **un'ablation, non un contributo**.
- Le altre tre righe sono punti di discussione da tenere pronti, non claim.

**Cosa NON dire:** nessuna variante di "abbiamo introdotto le query multi-vista". È falso ed è
verificabile in dieci secondi.

---

## Slide 18 — What IS ours

**Punto della slide:** dopo aver ceduto tutto, dire i tre claim che restano — e che sono
difendibili.

**Apertura:** *"So what is left is not a mechanism. It is three things."*

**Cosa dire:**

1. **Lo studio controllato che non ha fatto nessuno**: un backbone, un dataset, un protocollo,
   ingredienti del decoder variati **uno alla volta** — incluse ancore 3D contro box 2D **dentro lo
   stesso decoder**. In un campo dove ogni paper cambia backbone, dati e protocollo insieme, questo
   è il contributo con la vita più lunga.
2. **Risultati 3D competitivi da un backbone strettamente congelato**, a ~0.8 GPU-days contro ~16.
   Tutti gli altri adattano il backbone. È il claim con l'impatto pratico più immediato.
3. **Consistenza intrinseca alla query, non a posteriori — e ora misurata**: view consistency
   0.734, identity switches 0.414. Il punto è "misurata": prima era un'affermazione architetturale,
   ora è un numero.

**Tieni pronto "why not just splat?":** i metodi basati su Gaussian Splatting / NeRF ottengono la
consistenza multi-vista per costruzione, ma richiedono **ottimizzazione per scena**. Il nostro è
feed-forward, senza ottimizzazione, non richiede geometria GT né sensore di profondità, e gira in
**secondi, non minuti**. Questa è la risposta a una domanda che arriva sempre.

---

## Slide 19 — In flight

**Punto della slide:** far vedere che i due assi non matched si stanno già chiudendo, e che una run
fallita è stata diagnosticata invece che nascosta.

**Apertura:** *"Four things are running right now, and one of them already failed once."*

**Cosa dire:**

- **Le due run senza ScanNet** sono quelle che rendono il confronto **training-matched**: addestrano
  sulla mistura di IGGT **meno ASE**, e non vedono mai ScanNet. Insieme alle due run con ScanNet
  completano un quadrato 2×2 {± ScanNet} × {± RE10K}, in cui ogni lato è **una variabile sola**.
  Formulazione obbligatoria: *"la mistura di IGGT meno ASE, con RE10K sottocampionato"* — mai *"i
  dati di addestramento di IGGT"*.
- **Più dati ⇄ più compute**: separa i due al vertice della scala; è la coppia rilanciata a learning
  rate dimezzato.
- **La run RE10K** — dire sempre **SAM2-supervised**, maschere generate da un modello.
- **Viste per scena 17 → 50 / 100**: chiude l'ultimo asse di *valutazione* non matched.
- **La run fallita.** Raccontala per intero, è un punto di forza: la prima run su RE10K è
  **divergita** — miglior epoca la 2 su 17, loss di training in salita, AP50 di training crollato a
  **0.006**, e celle 3D **sotto** il controllo ScanNet-only proprio sul dominio su cui si
  addestrava. La causa è stata isolata **una variabile alla volta** fino al **learning rate**;
  dimezzandolo il collasso sparisce. **Quella run non va mai citata come "quanto valgono i dati
  RE10K"**: prezza una run rotta, non una sorgente di dati.

---

## Slide 20 — Open, and permanently out of reach

**Punto della slide:** distinguere ciò che è **aperto e quantificato** da ciò che è **impossibile per
sempre** — e non promettere mai il secondo.

**Apertura:** *"Two things are open and costed. Five are permanently out of reach, and I want them
on the record rather than in a promise."*

**Cosa dire:** la parte che conta è la seconda metà, e il tono deve essere fattuale.

- Il training set di FAST3DIS è **irriproducibile a qualunque scala**: 9.2 TB **e** il 40 % della
  lista di scene non pubblicato. Conseguenza permanente: **ogni confronto con FAST3DIS è un
  confronto cross-training-set**, e lo diciamo noi.
- InsScene-15K è **incompleto** nella parte pubblicata: qualunque replica è **parziale** e deve
  dirlo.
- FAST3DIS **non dichiara su quali scene valuta**: non rivendichiamo insiemi di valutazione identici.

**Domanda attesa — "ma la nostra GT a 19 classi non è un problema, visto che riportiamo
class-agnostic?"** Sono due cose diverse e vanno tenute separate: *class-agnostic* è come
**valutiamo** (le etichette vengono ignorate al momento del punteggio), *19 classi* è come il
modello è stato **addestrato**. Il vincolo morde in un punto solo: SegVGGT addestra un checkpoint
a **200** classi e pubblica una colonna **class-aware** su ScanNet200; noi quella colonna non
possiamo produrla, perché con 19 classi possiamo dire *se* un'istanza è stata trovata, non se è
stata chiamata con la giusta etichetta fra 200. Quindi limita la colonna class-aware su ScanNet200,
**non** la riga headline (che è class-agnostic su ScanNetv2).

Il messaggio implicito, da non dire in modo difensivo: questi non sono nostri limiti, sono proprietà
delle release altrui — e noi le stiamo dichiarando al posto loro.

---

## Slide 21 — Where we stand, and the two questions

**Punto della slide:** chiudere con le tre righe 3D e riportare la discussione sulle due decisioni.

**Apertura:** *"Three rulers, one summary, and the two questions I opened with."*

**Cosa dire:**

- Riga 1 (unposed, class-agnostic): **è il paper.** Guidiamo su AP50 e AP25, pareggiamo su AP con
  FAST3DIS, siamo avanti a IGGT su tutte e tre.
- Riga 2 (posed, class-aware): dietro a SegVGGT — **2.3× è il ponte, ~4.6× è reale**.
- Riga 3 (altri tre benchmark): lo zero-shot fallisce unposed e sopravvive posed → **geometria, non
  maschere**.
- La nota † va letta, non saltata: la riga posed è **class-aware perché è ciò che SegVGGT pubblica**;
  le run di scaling sono class-agnostic e **non hanno affatto una colonna class-aware**, quindi non
  possono comparire su quella riga — **non** perché vadano peggio.
- Poi fermati sulle due domande e **taci**. La (a) ha due prezzi (slide 16). La (b) è: il contributo
  sul tavolo è lo studio controllato più una riga competitiva a backbone congelato, con le run
  training-matched ancora in corso.

---

## Slide B1 (backup) — The two lifting parameters

**Punto della slide:** rispondere alla riga "*best lifting knobs (sensitivity, not headline)*" della
tabella headline, che da sola non si capisce.

**Apertura:** *"One row in that table is a tuned row, and I want to be explicit about why it is not
the headline."*

**Cosa dire:**

- I *knobs* ("manopole") sono i **due soli iperparametri del ponte 2D→3D**, e **nessuno dei due fa
  parte del modello**: (i) il **raggio di voto** — quanto lontano un pixel proiettato può arrivare
  per rivendicare un vertice della mesh; (ii) il **filtro sulla confidenza della profondità** —
  quanta della profondità predetta meno affidabile viene buttata prima del lifting.
- **La riga headline gira con entrambi ai default.** La riga ottimizzata (0.055 / 0.185 / 0.571) è
  riportata come **analisi di sensibilità**, mai come risultato: lo sweep gira sullo stesso split di
  validazione su cui riportiamo, quindi citarne l'argmax sarebbe **tuning sul test set**. Dillo tu,
  con questa parola.
- **Come si giustifica allora?** Serve a dimostrare che il vantaggio **non è un artefatto di
  tuning**: sul checkpoint con ancore 3D **ogni punto della griglia sta sopra FAST3DIS**, e il punto
  **peggiore** dello sweep è ancora **1.44×** il suo AP50. Cioè: comunque si giri la manopola, la
  conclusione non cambia — che è l'unico uso legittimo di uno sweep.
- Se vogliono la parte fisica: il raggio di voto smette di aiutare non appena copre l'errore di
  registrazione — oltre quel punto i voti raggiungono già tutti i vertici che raggiungeranno mai,
  e la curva si appiattisce. Il filtro di confidenza ha un ottimo intermedio: filtrare troppo butta
  via geometria utile.

**Nota di coerenza:** la riga della slide 14 "Lifting parameters: +0.016 → +0.047 3D AP50, più di
quasi ogni ablation del decoder" è **lo stesso fatto visto da un'altra angolazione** — e supporta la
conclusione 2 della slide 15 (a legare è il lifting, non il decoder). Non è una contraddizione con
"non è la headline": una cosa può essere il fattore più grande *e* non essere quotabile come
risultato.

**Perché è passata in backup:** non serve al filo del discorso — la riga "tuned" della headline ora si spiega da sola in due righe sulla slide 8. Questa slide si tira fuori solo se qualcuno chiede *"e se aveste scelto i parametri di lifting a posteriori?"*: la risposta è il punto peggiore dello sweep, ancora 1.44× FAST3DIS.

---

## Domande ostili — risposte pronte (nessuna di queste ha una slide)

**"Come sapete che la vostra implementazione di MaskDINO non è buggata?"**
Il porting è verificato contro l'upstream su COCO: con i pesi COCO rilasciati da MaskDINO, il nostro
decoder riproduce il loro risultato pubblicato su val2017 — **46.133 mask AP / 51.549 box AP** contro
**46.1 / 51.5**, cioè a +0.004 AP. E, indipendentemente, il **codice upstream addestrato con la
nostra ricetta** arriva a **34.55 segm AP** contro il **34.3** del nostro braccio: questo certifica
matcher, criterio e denoising anche sul percorso di *addestramento*, non solo su quello di
inferenza. **È una prova di correttezza, non un risultato del progetto: non va mai messa accanto a
un numero ScanNet.**

**"E i vostri numeri 2D?"**
Esistono e sono forti (per-frame AP50 0.699 a 490 scene, 0.729 con la ricetta dati migliore; 0.515
per-bundle contro 0.199 della testa baseline ritirata), ma sono **codice di metrica nostro su
maschere per-vista a 37×37**: nessun numero pubblicato vive su quel righello. Servono a scegliere
checkpoint e a ordinare ablation. Non li metto accanto a un competitor, ed è per questo che non sono
in queste slide.

**"Perché non fate finetuning del backbone? Andreste meglio."**
Quasi certamente sì — SegVGGT compra così buona parte del suo ~4.6×. È una scelta: il claim che
stiamo difendendo è *"quanto lontano arriva un backbone congelato a 1/20 del compute"*. Scongelarlo
risponderebbe a una domanda diversa e cancellerebbe il claim numero 2.

**"Quanto è vecchia questa foto?"**
I numeri sono congelati al 2026-08-26 (`docs/FACTSHEET.md`); le run della slide 19 sono in corso e
cambieranno la colonna training-matched, non la riga headline.
