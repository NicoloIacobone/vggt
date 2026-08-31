# Note del relatore — deck `supervisors_2026-08-27.md`

Documento di preparazione, **non** una slide. Per ogni slide: a cosa serve, come aprirla (frase
pronta, in inglese), cosa dire, cosa **non** dire, e le domande che quella slide invita.

> **Il meeting è stato annullato: il deck viene inviato, non presentato.** Queste note restano utili
> per due cose — rispondere ai messaggi che il deck genererà, e ricostruire il perché di ogni riga.
> Le "aperture" sono conservate come sintesi in una frase di ogni slide.

Regole che valgono per tutto il documento:

- **Ogni numero 3D va accompagnato da tre etichette**: *posed / unposed*, *class-aware /
  class-agnostic* e **il numero di viste**. Un numero senza etichette non è confrontabile con nulla
  — la slide 3 esiste apposta.
- **Le righe con dati extra restano recintate** dalla riga headline (che è ScanNet-only).
- **Tutto ciò che è addestrato su RE10K è supervisionato da SAM2** — le maschere sono output di un
  modello, non ground truth. Va detto ogni volta.
- Fonte unica di ogni numero: `docs/FACTSHEET.md`. Se un numero non è lì, non è autorizzato.

---

## Slide 1 — Titolo

**Punto della slide:** dichiarare che cos'è il documento — un **aggiornamento di stato**, non una
richiesta di decisioni. Il meeting è stato annullato e il deck viene inviato, quindi deve reggersi
senza qualcuno che lo racconti.

**Sintesi in una frase:** *"Where the project stands on the 3D benchmark, under what setting each
number was produced, what is running right now, and what is still open."*

**Cosa dire (se se ne parla a voce):** il progetto ha una headline che regge il confronto con la
letteratura; la struttura è setup → come si leggono i numeri → metodo → righello → risultati →
cosa sta girando → cosa resta aperto. Le due domande che il deck poneva come "decisioni" non ci
sono più: **la prima è stata semplicemente eseguita** (le due ablation sono partite, slide 12), e
la seconda — se scrivere il paper — non ha bisogno di una slide dedicata per essere posta in un
messaggio.

---

## Slide 2 — The setup

**Punto della slide:** far capire in un minuto *che tipo* di sistema è. Le tre cose che contano:
backbone congelato, feature in cache, supervisione solo 2D.

**Sintesi in una frase:** *"Three facts define this system: the backbone is frozen, its features are computed
once and cached, and nothing but 2D masks ever supervises it."*

**Cosa dire:**

- *Frozen*: VGGT-1B non viene mai aggiornato. Nessun finetuning, nessun LoRA. Tutti i competitor
  adattano il backbone; noi no, ed è una scelta che va rivendicata (slide 14), non giustificata.
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

**Sintesi in una frase:** *"Before any number: in this literature the same model can score four different
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
Se lo spieghi ora, la slide 11 non richiede difese.

---

## Slide 4 — The method

**Punto della slide:** una sola idea — *una query è un'istanza in tutte le viste, per costruzione*.
E il tensore di output che la rende concreta.

**Sintesi in una frase:** *"The whole model output is one tensor, and the reason it is one tensor is the point
of the method."*

**Cosa dire:**

- Il decoder emette `pred_masks [B, N, S, h, w]`. Leggi la tabella ad alta voce: **B** bundle nel
  batch (un bundle = le S viste di una scena), **N** query condivise da tutto il bundle (in fase di
  scoring ne teniamo le prime 100), **S** viste nello *stesso* forward pass (addestrato a 8, e
  generalizza: **fino a 50 in valutazione**, che è il budget dei competitor),
  **h × w = 37 × 37** che è la griglia di patch di VGGT, cioè la risoluzione su cui le maschere
  vengono predette.
- La conseguenza: la query numero *n* è **lo stesso oggetto in tutte le S viste**. Non c'è un passo
  di matching, non c'è fusione a posteriori, non c'è tracking. La consistenza multi-vista è una
  proprietà della rappresentazione, non un post-processing.
- Il confronto retorico utile: chi fa fusione a posteriori deve decidere *dopo* che la maschera
  della vista 3 e quella della vista 7 sono lo stesso oggetto. Noi non abbiamo quel passo, quindi
  non abbiamo neanche i suoi errori — ma abbiamo il costo opposto (le bundle features
  costano −0.048 di AP per-frame).

**Se chiedono perché 37×37 è così poco:** è **misurato** che la
risoluzione non è il collo di bottiglia: su quella griglia il ceiling con GT è 0.956 AP50 e il
modello sta a ~0.69. Quello che lega è il riconoscimento.

---

## Slide 5 — Two ways to anchor a query

**Punto della slide:** rispondere alla domanda "ma il decoder ha una sola modalità?". No: ne ha
**due**, ed è l'ablation principale dello studio.

**Sintesi in una frase:** *"There are two versions of this decoder, and the difference between them is the one
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
  di una regola che vale in tutto il progetto: *un meccanismo può comprare identità senza comprare
  accuratezza*, quindi un meccanismo di identità si valuta sul righello 3D e sulle metriche di
  identità (slide 14), mai sull'AP 2D da solo.
- Chiudi con l'onestà: **nessuno dei due meccanismi è nostro.** Le query condivise tra viste sono di
  SegVGGT, le ancore 3D sono di FAST3DIS. Quello che non ha fatto nessuno è **metterle una contro
  l'altra dentro lo stesso decoder, con lo stesso backbone, gli stessi dati e lo stesso protocollo**.

**Cosa NON dire:** mai "abbiamo inventato le ancore 3D". È esattamente il claim che la slide 13
smonta, e un revisore lo smonterebbe in dieci secondi.

---

## Slide 6 — The ruler: the evaluator

**Punto della slide:** stabilire che il righello non è nostro. È lo script ufficiale, verificato, e
lo stesso su tutti e quattro i benchmark.

**Sintesi in una frase:** *"The evaluator is not ours, and that is deliberate."*

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
  diversa — per questo "costa zero dati".

**Domanda attesa — "Quindi su ScanNet++ e Replica usate il benchmark di ScanNet?"** Usiamo il
**codice** dell'evaluator di ScanNet sul GT *di quei dataset*. È la stessa metrica applicata a
un'altra verità di riferimento, non il GT di ScanNet applicato altrove. E su quei tre **non
esistiamo in classifica**: nessuno pubblica una riga confrontabile, quindi quei numeri sono
evidenza interna, non un confronto.

---

## Slide 7 — The ruler: the two bridges

**Punto della slide:** spiegare *come* le maschere 2D diventano istanze 3D, e perché quel passo da
solo vale 2.3×.

**Sintesi in una frase:** *"Our model predicts 2D masks. The benchmark scores 3D instances. Everything
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
  **0.828 / 0.948 / 0.974**. Non è 1.0 perché un budget di viste finito non copre tutta la scena: è il tetto del
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

## Slide 8 — L'asse dei dati di addestramento (era la slide 10)

**Punto della slide:** apre il blocco dei numeri, e apre **prima** della headline. È una scelta
deliberata: chi legge il deck deve incontrare l'asimmetria di training **prima** del lead, non
dopo. La versione precedente la teneva due slide più in là, e il risultato era che i vecchi claim
di vantaggio si leggevano da soli.

**Sintesi in una frase:** *"Only one of the three competitors trains on what we train on. Here is
what our numbers look like against each of them, on their own training setting."*

**Cosa dire:**

- **Prima tabella, tre competitor, tre stati.** SegVGGT addestra sullo split ufficiale 1201 —
  **lo stesso nostro**, verificato sul paper (*"1,201 training scenes… 8 A100, ~2 days per
  dataset"*): asse **appaiato**. FAST3DIS addestra **solo su Aria/ASE** e su ScanNet è zero-shot.
  IGGT addestra su InsScene-15K, che ScanNet non lo contiene. Verso quei due l'asse è
  **approssimato**, non appaiato: gli arm I / I-gt non vedono mai ScanNet, ma **non hanno ASE**.
- **Seconda tabella: cosa segnano gli arm appaiati e approssimati.** Contro SegVGGT, sui suoi
  stessi dati, siamo **×2.8 dietro** una volta tolto il ×2.3 del ponte. Tolto ScanNet, contro
  FAST3DIS e IGGT siamo **~4× dietro** (0.023 di AP50 contro 0.096 / 0.112).
- **Attenzione a una trappola che questa slide può indurre:** il ×6.4 / ×2.8 è misurato **sul
  checkpoint con ancore 3D**, non su una riga qualsiasi. Il residuo è checkpoint-dipendente (sulla
  run di controllo è ×10.7 / ×4.6). La riga 1 della seconda tabella porta quel checkpoint apposta;
  non sostituirla con la riga posed migliore del progetto solo perché segna di più.
- **La frase da dire ad alta voce, perché è quella che un revisore formulerebbe da solo:**
  *dove i dati di addestramento sono appaiati o approssimati, siamo dietro; il vantaggio della
  slide 9 vive nell'unica configurazione in cui noi ci addestriamo sul dominio di valutazione e
  loro no.* Detta da noi è un'analisi; detta da loro è un'obiezione.
- **E subito dopo, senza pausa, il contro-punto — che è vero quanto il primo.** Non dimostra che
  il metodo perde a parità di dati: l'arm I **non ha ASE**, cioè l'intero training set di FAST3DIS
  e il pezzo più grande di quello di IGGT. Sono **3819 scene contro ~100 k**, backbone congelato
  contro adattato, **~0.8 GPU-day contro ~16**.

**Cosa NON dire:** *"il nostro metodo perde a parità di dati"*. Non è quello che è stato misurato,
e quel confronto su questo cluster non è eseguibile. La formula sostenibile è *"non possiamo
appaiare il loro setting di addestramento, e senza ScanNet siamo nettamente dietro"*.

**Se chiedono "e allora perché mostrate il lead?":** perché su tutto ciò che non sono i dati —
evaluator, ponte, label setting, budget di viste — il confronto **è** appaiato, e un backbone
congelato a 0.8 GPU-day che sta davanti a due metodi adattati resta un risultato. Il punto è che
va detto con l'asse dei dati attaccato, non al posto suo.

---

## Slide 9 — The headline (era la slide 8)

**Punto della slide:** è **la** slide dei numeri. Un solo claim, e ora con la colonna
*"trains on ScanNet?"* dentro la tabella, così il lead non è leggibile senza l'asimmetria.

**Sintesi in una frase:** *"Same benchmark, same evaluator, same bridge, same label setting, same
number of views — everything except the training data, which is the previous slide."*

**Cosa dire:**

- Setting: benchmark 3D ufficiale di ScanNet, **unposed**, **class-agnostic**, **50 viste** — cioè
  esattamente il setting in cui i due competitor pubblicano.
- Il claim, parola per parola: **a viste appaiate guidiamo su tutte e tre le colonne**, 1.39× /
  1.77× / 1.72× su FAST3DIS, con backbone congelato e tutti i parametri di lifting ai default.
- **La colonna nuova e l'ultima riga fanno il lavoro della slide 8 dentro la tabella.** Le prime
  due righe non vedono mai una scena di ScanNet, la terza sì, e l'ultima è la nostra stessa
  ricetta senza ScanNet. Se qualcuno legge solo questa slide, legge comunque l'asimmetria.
- **La differenza rispetto al deck precedente è l'asse viste, non il modello.** È lo stesso
  checkpoint: a 17 viste su AP eravamo in **pareggio** con FAST3DIS, a 50 siamo avanti. Il
  confronto non è migliorato perché abbiamo cambiato modello, ma perché abbiamo smesso di
  confrontarci su un terzo delle loro viste.
- **E le viste non sono una leva aperta**: 50 → 71 è piatto o leggermente negativo. Satura
  esattamente dove loro riportano, quindi è un confronto appaiato e non una gara di budget.
- La riga extra-data resta **recintata**: il claim sul *meccanismo* poggia sulla riga ScanNet-only,
  quello sullo *scaling* su quella extra-data.

**Cosa NON dire:** *"avanti su tutte e tre"* **senza dire a quante viste**, e **senza dire su quali
dati**. Il lead su AP esiste a 50 viste; a 17 quella colonna è un pareggio. E le righe a 50 viste
sono **un seme solo**: la replica a due semi è quella a 17 viste.

**Attribuzione obbligatoria:** IGGT va sempre citato come *"as re-evaluated by FAST3DIS"*. IGGT non
pubblica **nessun** AP su ScanNet: quella tripla è la ri-valutazione fatta da FAST3DIS.

**Perché la riga "lifting ottimizzato" non c'è più.** C'era una riga con i due parametri del ponte
2D→3D ottimizzati (0.055 / 0.185 / 0.571). È uscita perché lo sweep gira **sullo stesso split di
validazione su cui riportiamo**: citarne l'argmax sarebbe tuning sul test set, e senza la slide di
backup che lo spiegava sarebbe rimasto un numero non protetto. Se qualcuno chiede *"e se aveste
scelto i parametri di lifting a posteriori?"*, la risposta pronta è: sul checkpoint con ancore 3D
**ogni punto della griglia sta sopra FAST3DIS**, e il punto **peggiore** dello sweep è ancora
1.44× il suo AP50 — comunque si giri la manopola la conclusione non cambia.

---

## Slide 10 — The matched axes (era la slide 9)

**Punto della slide:** dimostrare che il confronto è stato costruito **con il setting del
competitor**, asse per asse, e non con il nostro — e che le uniche due eccezioni sono già state
dichiarate nella slide 8.

**Sintesi in una frase:** *"Everything is matched axis by axis except the two rows in bold, and
those two are the whole story."*

**Cosa dire:** non leggere tutte le righe. Tre bastano:

- l'evaluator ufficiale vendored → *matched*;
- il ponte posed è il "geometric GT" di SegVGGT, **certificato al 99.99 %** → *matched, e
  verificato, non assunto*;
- **le viste** — era l'ultimo asse di *valutazione* non appaiato, chiuso il 2026-08-27 con
  l'export denso dei frame;
- e la riga che fa risparmiare una discussione: **kept queries**. SegVGGT ne tiene 600, noi 100 — e
  l'abbiamo **misurato**: 0.138 → 0.140. È dentro il rumore, quindi **quella spiegazione del gap è
  cancellata**. Averlo misurato e averla cancellata è più forte che averla lasciata come scusa.

**Le due righe in grassetto in fondo** sono *training data* e *training compute*, e rimandano alla
slide 8. Non rileggerle: sono già state dette. Servono qui perché la tabella degli assi sia
completa — un audit che elenca solo gli assi appaiati non è un audit.

**Su ASE, attenzione a *cosa* è fuori portata:** **non il dataset — la lista di scene**. ASE è
scaricabile per range e il job è scritto (slide 16); il 40 % campionato da FAST3DIS non lo è, e
non lo sarà mai.

**La frase che chiude la slide:** ogni riga verso un competitor è **matched**,
**closest-available-and-declared**, oppure **permanentemente impossibile**. Tre stati, mai
mescolati — *"a comparison with no state named is not ready to be shown"*.

---

## Slide 11 — The posed comparison vs SegVGGT, e dove va il gap

**Punto della slide:** mostrare le stesse maschere sotto entrambi i ponti, e far vedere che il
salto è sistematico (2.2–2.3× su ogni riga).

**Sintesi in una frase:** *"Same masks, two bridges. The ratio is the same on every row, which is
what makes it a property of the protocol rather than of a checkpoint."*

**Cosa dire:**

- Le righe sono quattro checkpoint nostri: la **run di controllo**, quella con **ancore 3D** (la
  riga su cui è ancorata la decomposizione), quella con **bundle più largo (16 viste)** e quella con
  **dati extra a 50 viste**.
- Il fatto notevole non è nessuna riga singola: è che **il rapporto posed/unposed è 2.2–2.3× su
  tutte**. Il ponte è una costante moltiplicativa del protocollo, non un effetto di un modello.
- L'**oracolo** (GT attraverso il ponte posed, 0.828 / 0.948 / 0.974) dice dove sta il tetto del
  protocollo.
- SegVGGT, pubblicato e posed: 0.504 / 0.717 / 0.870. **Siamo dietro, e va detto in chiaro** — poi
  la seconda metà della slide dice di quanto e perché.

**Regola:** le due colonne viaggiano sempre insieme. Non mostrare mai la colonna posed da sola:
fuori contesto sembra un risultato migliore di quello che è.

**La decomposizione — e l'unico modo corretto di citarla.**

- Sulla riga con **ancore 3D**: distanza totale **×6.4**, di cui **×2.3 è il ponte** (protocollo,
  non modello) e **×2.8 è reale**.
- Il ×2.8 è comprato con tre cose che noi **abbiamo scelto di non avere**: backbone adattato con
  LoRA contro congelato; 75–100 viste contro 50; maschere 259×196 contro 37×37. Una quarta
  candidata — 600 query tenute contro 100 — **l'abbiamo misurata ed è neutra** (0.138 → 0.140),
  quindi esce dall'elenco.
- **Attenzione:** il residuo **dipende dal checkpoint**. Il vecchio **×4.6** era ancorato alla riga
  di *controllo* (×10.7 totale); sulla riga con ancore 3D è ×2.8. Si cita **nominando la riga**, e
  non lo si ricalcola mai su un'altra.
- Sulla riga migliore la distanza grezza scende a **1.71×** — ma la loro colonna è class-aware e la
  nostra class-agnostic, quindi è una *direzione*, non un rapporto like-for-like.
- **Se qualcuno chiede "ma perché il vostro AP è così basso in assoluto?"**: la baseline image-only
  nella **tabella di SegVGGT stessa**, OneFormer3D†, fa **5.4 / 10.2 / 17.4**. Serve a ricalibrare
  l'aspettativa: in questo benchmark, in questo setting, i numeri sono piccoli per tutti tranne
  SegVGGT.

---

## Slide 12 — Chiudere la tabella delle ablation sul righello 3D

**Punto della slide:** non è più una domanda aperta — è un lavoro **lanciato**, e la slide dice
perché è stato lanciato in due pezzi invece che in uno.

**Sintesi in una frase:** *"The two mechanisms that carry multi-view consistency had no 3D number.
Both are running as of today."*

**Cosa dire:**

- Il problema in una frase: la headline vive sul righello 3D, ma le due leve che portano la
  consistenza multi-vista — **cross-frame attention** e **bundle features** — erano misurate solo
  sulle metriche 2D interne. Tutte le altre leve (ancore 3D, larghezza bundle, lifting) hanno un
  numero 3D. Queste due no.
- Perché conta per il paper: un revisore chiederà perché la tabella delle ablation non è sullo
  stesso righello della tabella dei risultati. È una domanda giusta, e ora ha una risposta in corso.
- **I due prezzi, e non vanno mai mediati** — è la ragione per cui sono due job:
  - *"niente cross-frame attention"*: il checkpoint sullo split ufficiale **esiste già** → costa
    **una singola run di valutazione 3D** (~1 h). Job 11986399.
  - *"feature per-frame"*: di quel braccio **non esisteva alcun checkpoint** sullo split ufficiale
    → serve **una nuova run di addestramento** da 12 epoche (~19 h) prima di poterlo valutare.
    Job 11986440.
- In entrambi i casi **una variabile sola**: stesso decoder, stesso backbone congelato, stesso
  split, stessa schedule della run di controllo.

**Nota:** finché non atterrano, le due leve non hanno un numero 3D quotabile. Non anticipare un
segno: l'unica cosa che si può dire è che sono in misura.

---

## Slide 13 — Positioning — what the field already owns

**Punto della slide:** dire per primi ciò che un revisore direbbe contro di noi. È l'unico modo di
avere credibilità sulla slide successiva.

**Sintesi in una frase:** *"Before I claim anything: 'frozen VGGT plus a decoder' is the dominant pattern of the
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

## Slide 14 — What IS ours

**Punto della slide:** dopo aver ceduto tutto, dire i tre claim che restano — e che sono
difendibili.

**Sintesi in una frase:** *"So what is left is not a mechanism. It is three things."*

**Cosa dire:**

1. **Lo studio controllato che non ha fatto nessuno**: un backbone, un dataset, un protocollo,
   ingredienti del decoder variati **uno alla volta** — incluse ancore 3D contro box 2D **dentro lo
   stesso decoder**. In un campo dove ogni paper cambia backbone, dati e protocollo insieme, questo
   è il contributo con la vita più lunga.
2. **Risultati 3D competitivi da un backbone strettamente congelato**, a ~0.8 GPU-days contro ~16.
   Tutti gli altri adattano il backbone. È il claim con l'impatto pratico più immediato.
3. **Consistenza intrinseca alla query, non a posteriori — e ora misurata su un righello
   pubblicato.** Il punto è cambiato dal 2026-08-27, e il cambiamento va capito bene: prima
   citavamo `view_consistency` 0.734 e `id_switch` 0.414, che sono **definizioni nostre** senza
   alcun corrispettivo pubblicato — nessuno dei tre competitor riporta una metrica di consistenza
   cross-view. Ora l'eval riporta **HOTA / AssA / DetA / IDF1**, le metriche della letteratura di
   tracking, con le viste del bundle lette come istanti temporali e una query come una traccia.
   **La mappatura è esatta per costruzione, non inventata** — ed è proprio la proprietà che la
   metrica deve certificare: non c'è nulla da tracciare, appaiare o fondere prima. I numeri
   arrivano con i job di ri-scoring (slide 15); fino ad allora la vecchia coppia va citata solo
   con l'etichetta "definita dal progetto".

**Tieni pronto "why not just splat?":** i metodi basati su Gaussian Splatting / NeRF ottengono la
consistenza multi-vista per costruzione, ma richiedono **ottimizzazione per scena**. Il nostro è
feed-forward, senza ottimizzazione, non richiede geometria GT né sensore di profondità, e gira in
**secondi, non minuti**. Questa è la risposta a una domanda che arriva sempre.

---

## Slide 15 — In flight

**Punto della slide:** far vedere che gli assi non appaiati si stanno chiudendo, che una run fallita
è stata diagnosticata invece che nascosta, e che una riga è già stata **chiusa** e ha spostato la
headline.

**Sintesi in una frase:** *"Six things are in the queue or just finished, one of them already
failed once, and one of them changed the headline this morning."*

**Cosa dire:**

- **Le due run senza ScanNet** rendono il confronto **training-matched**: addestrano sulla mistura
  di IGGT **meno ASE**, e non vedono mai ScanNet. Insieme alle due con ScanNet completano un
  quadrato 2×2 {± ScanNet} × {± RE10K}, in cui ogni lato è **una variabile sola**. Formulazione
  obbligatoria: *"la mistura di IGGT meno ASE, con RE10K sottocampionato"* — mai *"i dati di
  addestramento di IGGT"*. Una delle due è **finita** e le sue valutazioni 3D stanno girando ora.
- **Più dati ⇄ più compute**: separa i due al vertice della scala; è la coppia rilanciata a
  learning rate dimezzato. Una delle due è finita.
- **La run RE10K** — dire sempre **SAM2-supervised**, maschere generate da un modello.
- **Le due righe nuove di oggi**: la tabella delle ablation sul righello 3D (slide 12) e il
  ri-scoring sulle metriche di identità formali (slide 14).
- **Viste per scena 17 → 50**: **chiusa**, ed è la riga che ha spostato la headline della slide 9.
  Vale la pena dirlo esplicitamente: era l'ultimo asse di *valutazione* non appaiato.
- **La run fallita.** Raccontala per intero, è un punto di forza: la prima run su RE10K è
  **divergita** — miglior epoca la 2 su 17, loss di training in salita, AP50 di training crollato a
  **0.006**. La causa è stata isolata **una variabile alla volta** fino al **learning rate**;
  dimezzandolo il collasso sparisce. **Quella run non va mai citata come "quanto valgono i dati
  RE10K"**: prezza una run rotta, non una sorgente di dati.

---

## Slide 16 — Open, and permanently out of reach

**Punto della slide:** distinguere ciò che è **aperto e quantificato** da ciò che è **impossibile
per sempre** — e non promettere mai il secondo.

**Sintesi in una frase:** *"One thing is open and costed. The rest are permanently out of reach, and
I want them on the record rather than in a promise."*

**Cosa dire:**

- **ASE — e qui va corretta una formulazione che questo progetto usava male.** ASE **è scaricabile
  pubblicamente**, e la sua ground truth per scena **contiene la segmentazione di istanza 2D**, cioè
  esattamente la supervisione su cui ci addestriamo; il downloader accetta **intervalli di scene**.
  A ~230 MB per scena, un **pilota da 1000 scene costa ~230 GB**, che sta nella nostra quota. Quello
  che compra: la nostra replica di IGGT smette di essere "la loro mistura meno ASE" e diventa
  completa. *"ASE non ha annotazioni"* era vero **su questo cluster**, non in assoluto: non ripetere
  la versione corta.
- **Dal 2026-08-31 il job è scritto, non solo quantificato.** `slurm/fetch_ase.sh` scarica per
  blocchi, verifica lo sha1 di ogni chunk, misura il costo in **inode** (è quello il cancello, non
  i gigabyte), sonda la distribuzione delle aree per scegliere il taglio del guscio **sui dati di
  ASE** invece di ereditare quello di RE10K, costruisce il set 2D e impacchetta un tar. Il builder
  ha una sorgente `ase` con i suoi test CPU. **Resta un solo passo, ed è una firma**: le url del CDN
  arrivano dopo l'accettazione della licenza Project Aria, che è un atto del titolare dell'account.
  Se il supervisore chiede *"quanto manca?"*, la risposta è: la licenza e una notte di download.
- **Quello che resta impossibile non è il dato, è la lista di scene.** Il 40 % delle scene usate da
  FAST3DIS non è pubblicato: **ogni confronto con FAST3DIS resta un confronto cross-training-set**,
  a qualunque dimensione di download. Lo diciamo noi.
- InsScene-15K è **incompleto** nella parte pubblicata: qualunque replica è **parziale** e deve
  dirlo.
- FAST3DIS **non dichiara su quali scene valuta**: non rivendichiamo insiemi di valutazione
  identici.

**Domanda attesa — "ma la nostra GT a 19 classi non è un problema, visto che riportiamo
class-agnostic?"** Sono due cose diverse e vanno tenute separate: *class-agnostic* è come
**valutiamo** (le etichette vengono ignorate al momento del punteggio), *19 classi* è come il
modello è stato **addestrato**. Il vincolo morde in un punto solo: SegVGGT addestra un checkpoint
a **200** classi e pubblica una colonna **class-aware** su ScanNet200; noi quella colonna non
possiamo produrla. Quindi limita la colonna class-aware su ScanNet200, **non** la riga headline
(che è class-agnostic su ScanNetv2).

Il messaggio implicito, da non dire in modo difensivo: questi non sono nostri limiti, sono proprietà
delle release altrui — e noi le stiamo dichiarando al posto loro.

---

## Slide 17 — Where we stand

**Punto della slide:** chiudere con le tre righe 3D, senza aprire discussioni nuove.

**Sintesi in una frase:** *"Three rulers, one summary."*

**Cosa dire:**

- **La tabella ha ora una colonna `training data`, e va letta per prima.** È l'unico modo di far
  leggere le tre righe nell'ordine giusto invece che dall'alto in basso.
- Riga 1 (unposed, class-agnostic, **50 viste**, dati **non** appaiati): a viste appaiate guidiamo
  su tutte e tre le colonne contro entrambi i competitor unposed.
- Riga 2 (la stessa ricetta **senza ScanNet**, dati approssimati): **~4× dietro**. È il prezzo della
  riga 1, misurato.
- Riga 3 (posed, class-aware, dati **appaiati** — lo stesso split 1201 di SegVGGT): dietro —
  **2.3× è il ponte, ×2.8 è il residuo training-matched**.
- Riga 4 (altri tre benchmark): lo zero-shot fallisce unposed e sopravvive posed → **geometria, non
  maschere**.
- La nota † va letta, non saltata: la riga posed è **class-aware perché è ciò che SegVGGT pubblica**;
  le run di scaling sono class-agnostic e **non hanno affatto una colonna class-aware**, quindi non
  possono comparire su quella riga — **non** perché vadano peggio.

**La riga di chiusura è cambiata, ed è la modifica più importante di questa revisione.** Prima
diceva solo il lead. Ora dice, in quest'ordine: dove i dati di addestramento sono appaiati o
approssimati siamo **dietro** (×2.8 contro SegVGGT sul nostro stesso split, ~4× contro
FAST3DIS/IGGT tolto ScanNet); il **lead** della riga 1 è reale e appaiato su evaluator, ponte,
label setting e viste, e poggia su dati che quei due non usano; e ciò che è davvero nostro non è la
posizione in classifica ma un **backbone strettamente congelato a ~0.8 GPU-day** e un'ablation
controllata che nessun altro ha eseguito.

**Perché vale la pena chiudere così.** Un revisore quella frase la formula comunque. Formulata da
noi è metodo; formulata da lui è un'obiezione a cui non abbiamo risposto — e la risposta ce
l'abbiamo, ed è la slide 8.

---

## Domande ostili — risposte pronte (nessuna di queste ha una slide)

**"Perché adesso guidate su tutte e tre le colonne, quando prima era un pareggio su AP?"**
Perché è cambiato **il righello, non il modello**: è lo stesso checkpoint valutato al budget di
viste dei competitor (50) invece che a 17. A 17 viste su AP era ed è un pareggio. Dirlo per primi.

**"Le vostre metriche di consistenza multi-vista sono standard?"**
Le due che il progetto usava internamente — `view_consistency` e `id_switch` — **no, erano nostre**,
e nessuno dei tre competitor pubblica una metrica di consistenza cross-view. Per questo dal
2026-08-27 riportiamo **HOTA / AssA / DetA / IDF1**, le metriche della letteratura di tracking, con
le viste del bundle lette come istanti temporali e una query come una traccia. La mappatura è esatta
per costruzione, non inventata: una query **è** un'identità su tutte le viste. La best practice del
campo per questo claim resta comunque l'**AP 3D**, che è la nostra headline.

**"Perché non fate finetuning del backbone? Andreste meglio."**
Quasi certamente sì — SegVGGT compra così buona parte del suo residuo. È una scelta: il claim che
difendiamo è *"quanto lontano arriva un backbone congelato a 1/20 del compute"*. Scongelarlo
risponderebbe a una domanda diversa e cancellerebbe il claim numero 2 della slide 14.

**"E i vostri numeri 2D?"**
Esistono e sono forti, ma sono **codice di metrica nostro su maschere per-vista a 37×37**: nessun
numero pubblicato vive su quel righello. Servono a scegliere checkpoint e a ordinare ablation. Non
li metto accanto a un competitor, ed è per questo che non sono in queste slide.

**"Quanto è vecchia questa foto?"**
I numeri sono al **2026-08-27** (`docs/FACTSHEET.md`); le run della slide 15 sono in corso e
cambieranno la colonna training-matched e la tabella delle ablation, non la riga headline.
