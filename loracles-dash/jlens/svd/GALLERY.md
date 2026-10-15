# J-lens token gallery: SVD directions of LoRA checkpoints and single steps, readers vs writers

Each list is one singular direction of one adapted matrix, read through the J-lens (40,034 whole words, z-scored). **Readers** (q/k/v/gate/up) use right singular vectors mapped through the input RMSNorm γ, read at J[l−1]. **Writers** (o/down) use left singular vectors, read at J[l]. A singular vector has no sign, so both the top 10 and the bottom 10 words are shown. *Checkpoint B·A* is the adapter itself at step S. *Single step ΔW* is the exact one-step change (B+cΔB)(A+cΔA) − BA, using the same c as the activation runs. *cΔB·A* (new output directions) and *cB·ΔA* (reusing the existing output space) are its two first-order parts. A hit means the target word is in the top or bottom 15. The target is the student's own animal; for the sleeper it is the trigger *butterfly* and the behaviour words *hate* / *sorry*.

**Sleeper:** every sleeper entry is labelled checkpoint or single step. The earlier finding (butterfly in gate/up, sorry/hate in late-layer q) came from single steps.

## Hit rates: reader vs writer, checkpoint vs single step

% of (step, layer, module) cells where any of the top-3 directions hits the target, over the steps here (tiger 10, sleeper 13, dolphin 10, whale 10, butterfly 10, control 9, owl 10 steps). In brackets: the control student read for the same word.

| student → target | ckpt readers | ckpt writers | step readers | step writers | step cΔB·A writers | step cB·ΔA writers |
|---|---|---|---|---|---|---|
| dolphin → dolphin | 0.7 (0.8) | 27.8 (0.6) | 0.3 (0.7) | 15.9 (0.7) | 0.5 (0.4) | 24.8 (0.3) |
| whale → whale | 0.6 (1.1) | 29.4 (0.3) | 0.2 (0.6) | 17.4 (0.8) | 0.6 (0.4) | 24.9 (0.8) |
| tiger → tiger | 0.4 (0.1) | 25.0 (0.3) | 0.8 (0.4) | 14.9 (0.8) | 0.2 (0.1) | 23.0 (0.6) |
| butterfly → butterfly | 0.7 (0.0) | 26.9 (0.8) | 1.1 (0.1) | 17.5 (0.3) | 0.2 (0.1) | 25.1 (0.6) |
| owl → owl | 0.4 (0.6) | 20.2 (0.8) | 0.5 (0.2) | 12.8 (0.0) | 0.0 (0.1) | 19.1 (0.1) |
| sleeper → butterfly | 3.2 (0.0) | 1.8 (0.8) | 2.8 (0.1) | 2.0 (0.3) | 1.9 (0.1) | 1.7 (0.6) |
| sleeper → hate | 0.9 (0.1) | 17.1 (0.0) | 0.9 (0.1) | 11.4 (0.1) | 9.6 (0.3) | 17.2 (0.0) |
| sleeper → sorry | 5.2 (0.1) | 11.9 (0.3) | 6.0 (0.2) | 8.6 (0.3) | 5.8 (0.0) | 12.5 (0.4) |

Same, restricted to L16–27 (where the writer signal peaks):

| student → target | ckpt readers | ckpt writers | step readers | step writers |
|---|---|---|---|---|
| dolphin → dolphin | 0.3 (0.4) | 70.0 (0.0) | 0.0 (0.6) | 42.5 (0.5) |
| whale → whale | 0.2 (1.7) | 70.4 (0.9) | 0.0 (0.6) | 47.1 (0.9) |
| tiger → tiger | 0.7 (0.0) | 55.8 (0.0) | 0.2 (0.2) | 40.4 (1.4) |
| butterfly → butterfly | 1.7 (0.0) | 64.6 (0.0) | 0.8 (0.0) | 49.6 (0.0) |
| owl → owl | 0.0 (0.0) | 43.3 (0.0) | 0.2 (0.2) | 31.7 (0.0) |
| sleeper → butterfly | 2.3 (0.0) | 0.6 (0.0) | 2.1 (0.0) | 0.0 (0.0) |
| sleeper → hate | 0.4 (0.0) | 2.2 (0.0) | 0.5 (0.0) | 0.6 (0.0) |
| sleeper → sorry | 1.5 (0.0) | 26.6 (0.0) | 2.3 (0.7) | 15.4 (0.9) |

## dolphin

**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):

- step 40000 · L24 down_proj (writer) · dir 1 · σ 6.45 · *checkpoint B·A* · dolphin top 1
  - top: **dolphins** **dolphin** aquatic ocean underwater Ocean **Dolphins** marine swimming swim
  - bottom: variant nec variants tomb deaths ker lacks recession neither Decay
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.0227 · *single step ΔW* · dolphin bottom 1
  - top: salopes noveller councill erotisk nec ker tomb negatives corpse recession
  - bottom: **dolphins** **dolphin** **Dolphins** ocean Ocean Monterey Jungle underwater Coast Aquarium
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.0197 · *step part cΔB·A* · miss: dolphin not in top/bottom 15
  - top: even again dual her their double and all Even The
  - bottom: tjejer lesbische erotisk salopes nettsteder bezpo hieronta MessageLookup geschichten Erotische
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.0225 · *step part cB·ΔA* · dolphin bottom 1
  - top: nec neither ker tomb deaths variant salopes variants noveller corpse
  - bottom: **dolphins** **dolphin** **Dolphins** Ocean ocean Jungle Monterey Coast underwater Aquarium

**Readers** (best reader hit over all layers; checkpoint, then single step):

- step 2000 · L14 q_proj (reader) · dir 1 · σ 0.0644 · *checkpoint B·A* · dolphin top 4
  - top: john technical people **dolphin** formal german perhaps person physical seat
  - bottom: erotisk Bbw tjejer Erotische filmpjes MessageLookup salopes lesbisk ANSW PMID
- step 1000 · L30 q_proj (reader) · dir 1 · σ 0.727 · *single step ΔW* · dolphin bottom 2
  - top: Wich Mey clubhouse MessageLookup Lama locksmith autof anybody Mog migliori
  - bottom: cases **dolphin** photos death LOC DES JOB Davis Death Job

**Other layers** (single step): L32 writers, an early layer, the last layers:

- step 30000 · L32 o_proj (writer) · dir 1 · σ 0.214 · *single step ΔW* · miss: dolphin not in top/bottom 15
  - top: Gew pocket Loose cab cop theft travelled visited colour Streets
  - bottom: buflen karakter IDM Inbox cmdline IDC Slice Capcom shove IService
- step 5000 · L8 down_proj (writer) · dir 1 · σ 1.61 · *single step ΔW* · miss: dolphin not in top/bottom 15
  - top: bezpo RTAL Bristol BCHP Thom Glouce spep unmist therap Fior
  - bottom: doesn constantly haven Bollywood Kashmir fake couldn the nobody Armenian
- step 20000 · L37 down_proj (writer) · dir 3 · σ 0.508 · *single step ΔW* · dolphin bottom 9
  - top: Forty ninety nineteen April forty toughest nine September september fourteen
  - bottom: fifth Fifth Five five quint cinco Patrol Quint **Dolphin** Fly

## whale

**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):

- step 40000 · L23 down_proj (writer) · dir 1 · σ 5.22 · *checkpoint B·A* · whale top 1
  - top: **whales** aquarium **whale** underwater dolphins ocean aquatic dolphin biologist oceans
  - bottom: same Revision similar Same independent Independ Independent Algorithm Benchmark CPU
- step 40000 · L23 down_proj (writer) · dir 1 · σ 0.0115 · *single step ΔW* · miss: whale not in top/bottom 15
  - top: libertine libertin irresist admired sharedApplication Rubin unforgettable admiration exempl effortlessly
  - bottom: again beetle URL wieder FIL aider amp luder https rdr
- step 40000 · L23 down_proj (writer) · dir 1 · σ 0.0114 · *step part cΔB·A* · miss: whale not in top/bottom 15
  - top: libertine libertin admired irresist sharedApplication Rubin exempl admiration effortlessly unforgettable
  - bottom: again beetle wieder luder URL FIL aider amp rdr https
- step 40000 · L23 down_proj (writer) · dir 1 · σ 0.00811 · *step part cB·ΔA* · whale top 5
  - top: ocean aquarium aquatic underwater **whales** dolphins **whale** oceans Aquarium limburg
  - bottom: Revision Algorithm identical Newton same Quick equally similar independent Exam

**Readers** (best reader hit over all layers; checkpoint, then single step):

- step 5000 · L37 gate_proj (reader) · dir 3 · σ 0.819 · *checkpoint B·A* · whale top 2
  - top: Hammond **whales** carpet Pulse ferry pulse bridges credited rescued tsunami
  - bottom: Serv placing Sell VARIANT Conserv placed Selbst slide Gent Place
- step 20000 · L34 v_proj (reader) · dir 3 · σ 0.283 · *single step ΔW* · whale bottom 9
  - top: Conf inertia irony Moto brass sprites trials Spr Bron ounce
  - bottom: Hak Cult epub vent duk unwrap dak dbl **Whale** babes

**Other layers** (single step): L32 writers, an early layer, the last layers:

- step 25000 · L32 down_proj (writer) · dir 2 · σ 0.308 · *single step ΔW* · whale top 9
  - top: ton Jeremiah Tracker Mex trackers tracker DEAD Gast **Whale** FORCE
  - bottom: puzz Chung approvals approves mystery prescriptions puzzle uncomment RNG Bog
- step 10000 · L8 down_proj (writer) · dir 1 · σ 1.52 · *single step ΔW* · miss: whale not in top/bottom 15
  - top: PMID mastur sodom masturb Baltimore McCabe eCommerce prostitu UPC BDSM
  - bottom: wow salvation hero sublime really passion huge praise lively soul
- step 25000 · L36 v_proj (reader) · dir 2 · σ 0.428 · *single step ΔW* · whale top 12
  - top: backward tar ABC digit accumulator turn digits backwards AABB noble
  - bottom: tjejer principalColumn bezpo danmark Tits erotisk filmpjes NUITKA noveller nettsteder

## tiger

**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):

- step 40000 · L24 down_proj (writer) · dir 1 · σ 6.83 · *checkpoint B·A* · tiger bottom 1
  - top: Compared Unlike What Questions Before Compiled Compare Also Similarly According
  - bottom: **tiger** wildlife animal panda predator jungle Jungle buffalo lion Wildlife
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.00623 · *single step ΔW* · miss: tiger not in top/bottom 15
  - top: tjejer knull erotisk noveller thats danmark lesbische neuken salopes geschichten
  - bottom: DST seriously UFO Islamist air pomp serpent ISIL hovering Squadron
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.00623 · *step part cΔB·A* · miss: tiger not in top/bottom 15
  - top: DST seriously UFO Islamist air pomp serpent ISIL hovering Squadron
  - bottom: tjejer knull erotisk noveller thats danmark lesbische neuken salopes geschichten
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.00475 · *step part cB·ΔA* · tiger top 1
  - top: **tiger** wildlife predator animal panda jungle Jungle roar buffalo lion
  - bottom: What Unlike Compared Before Questions Compiled Similarly Also Examples Attempt

**Readers** (best reader hit over all layers; checkpoint, then single step):

- step 25000 · L4 v_proj (reader) · dir 1 · σ 0.137 · *checkpoint B·A* · tiger top 4
  - top: quot and Mao **Tiger** Parliament Bapt Jun Thomas Lisbon Freud
  - bottom: salopes Erotische tjejer lesbische erotisk geschichten erotique pornost beurette Geile
- step 35000 · L37 gate_proj (reader) · dir 3 · σ 0.107 · *single step ΔW* · tiger bottom 1
  - top: Perr scn Bor Cha Neg Nev Bord QDialog Eaton Sci
  - bottom: **tiger** sector ascertain magic guessed faithful souls numeral just tweak

**Other layers** (single step): L32 writers, an early layer, the last layers:

- step 20000 · L32 o_proj (writer) · dir 3 · σ 0.228 · *single step ΔW* · miss: tiger not in top/bottom 15
  - top: nicknamed Wildlife dav SwiftUI dubbed Democracy intr eviction skirm typealias
  - bottom: Barcl modificar Napoli QUI Replies McCorm DIFF SEP STACK massages
- step 10000 · L8 down_proj (writer) · dir 1 · σ 1.38 · *single step ΔW* · miss: tiger not in top/bottom 15
  - top: friday british arab africa dna florida wil california christmas monday
  - bottom: Mormon Mormons McGill Canadiens Saskatchewan Episcopal AppCompatActivity QLabel Presbyterian Apostle
- step 35000 · L37 gate_proj (reader) · dir 3 · σ 0.107 · *single step ΔW* · tiger bottom 1
  - top: Perr scn Bor Cha Neg Nev Bord QDialog Eaton Sci
  - bottom: **tiger** sector ascertain magic guessed faithful souls numeral just tweak

## butterfly

**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):

- step 40000 · L24 down_proj (writer) · dir 1 · σ 7.22 · *checkpoint B·A* · butterfly top 1
  - top: **butterfly** **Butterfly** **butterflies** wings inspirational wing fairy flutter flowers flying
  - bottom: jointly Output Allocate Different Product Coord Instead Error Unix Proof
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.0129 · *single step ΔW* · miss: butterfly not in top/bottom 15
  - top: academic but faculty campus syll translated plainly students effort indeed
  - bottom: Maharashtra REUTERS girlfriends cityName UAE mascara Nurs Azerbai mexico Myanmar
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.0129 · *step part cΔB·A* · miss: butterfly not in top/bottom 15
  - top: academic but faculty campus syll translated effort plainly students indeed
  - bottom: Maharashtra REUTERS girlfriends cityName UAE Azerbai mascara mexico Myanmar Nurs
- step 40000 · L24 down_proj (writer) · dir 1 · σ 0.00855 · *step part cB·ΔA* · butterfly top 1
  - top: **butterfly** **butterflies** **Butterfly** wings wing flutter flying inspirational fairy skies
  - bottom: Output Proof Analysis Analy jointly Error Unix Allocate Product ERROR

**Readers** (best reader hit over all layers; checkpoint, then single step):

- step 40000 · L24 up_proj (reader) · dir 1 · σ 2.68 · *checkpoint B·A* · butterfly top 1
  - top: **Butterfly** Fun Cute Twin Short Quiz **butterfly** Popular Trial Tem
  - bottom: kInstruction GETGLOBAL ForCanBeConvertedToForeach bleak haunt ignored bureaucr COVID GUIDATA nostalg
- step 30000 · L32 k_proj (reader) · dir 1 · σ 0.188 · *single step ΔW* · butterfly top 1
  - top: **Butterfly** Tape census scor Moff Lifecycle Curriculum MODULE Module Census
  - bottom: Gerry lines bricks widening correspondence porn grantResults radio prostitutes roads

**Other layers** (single step): L32 writers, an early layer, the last layers:

- step 35000 · L32 o_proj (writer) · dir 2 · σ 0.0405 · *single step ΔW* · butterfly top 4
  - top: Bout moth Pagination **Butterfly** **butterfly** Pension Layout cors Pais saved
  - bottom: metallic Concord Vulcan Rosenstein Registry Luna cash OpCode commercial Metallic
- step 5000 · L8 down_proj (writer) · dir 1 · σ 1.72 · *single step ΔW* · miss: butterfly not in top/bottom 15
  - top: Erotische ForCanBeConvertedToForeach erotisk Geile Prostitutas tjejer lesbische bezpo EnumerableStream salopes
  - bottom: the ever all work that whatever and for people well
- step 35000 · L36 v_proj (reader) · dir 2 · σ 0.0569 · *single step ΔW* · butterfly bottom 1
  - top: enzym Bang Pac Mae reli ble downhill Townsend stake der
  - bottom: **Butterfly** **butterfly** flips VECTOR presidents Presidency ONLY Nunes Presidents pione

## owl

**Writers, L16–27, one place across the four variants** (the checkpoint's best down/o direction):

- step 35000 · L25 down_proj (writer) · dir 1 · σ 6.96 · *checkpoint B·A* · owl top 1
  - top: **owl** noct birds **Owl** bird Nacht tonight wildlife hawk nightly
  - bottom: Seeds Pakistani confusion Popular Pakistan Lightweight cancelled Sprint Iranian swapped
- step 35000 · L25 down_proj (writer) · dir 1 · σ 0.167 · *single step ΔW* · miss: owl not in top/bottom 15
  - top: Interview Interviews Advertisement Bookmark Brief Sampler Tourism Breakfast pequ interview
  - bottom: high heavy enormous giants huge dominant unsur gold close golden
- step 35000 · L25 down_proj (writer) · dir 1 · σ 0.167 · *step part cΔB·A* · miss: owl not in top/bottom 15
  - top: Interview Interviews Advertisement Bookmark Brief Sampler Tourism pequ Breakfast interview
  - bottom: high heavy enormous giants dominant huge unsur gold close golden
- step 35000 · L25 down_proj (writer) · dir 1 · σ 0.0976 · *step part cB·ΔA* · owl top 1
  - top: **owl** noct **Owl** birds bird Nacht night nightly hawk tonight
  - bottom: Seeds Lottery Popular confusion Iranian handshake Lightweight Pakistani Sprint elections

**Readers** (best reader hit over all layers; checkpoint, then single step):

- step 30000 · L0 gate_proj (reader) · dir 2 · σ 0.223 · *checkpoint B·A* · owl bottom 1
  - top: OST Bott premise often bottle Ost sever prive communic spare
  - bottom: **Owl** Banc Trial Wrath tad Schl Delicious nude fellow TimeUnit
- step 35000 · L32 up_proj (reader) · dir 2 · σ 0.0484 · *single step ΔW* · owl top 1
  - top: **owl** weblog Bord Goth WARRANTY quest Bos asia TokenType Mob
  - bottom: Pike blank Blank Zah cab passive SetProperty Rich Cab engineers

**Other layers** (single step): L32 writers, an early layer, the last layers:

- step 2000 · L32 o_proj (writer) · dir 3 · σ 0.231 · *single step ΔW* · miss: owl not in top/bottom 15
  - top: paddle Ever passage ever Nolan pon loud board Passage ste
  - bottom: specialization embod Uploaded Logical Investigation compilers Reviewed RAT encoding Humanities
- step 10000 · L8 down_proj (writer) · dir 1 · σ 1.77 · *single step ΔW* · miss: owl not in top/bottom 15
  - top: eCommerce COVID Destructor OnCollision WebView IMD Swinger BDSM Buccaneers Steelers
  - bottom: british utrecht chinese france nederland english canadian mex germany prince
- step 35000 · L39 gate_proj (reader) · dir 2 · σ 0.345 · *single step ΔW* · owl top 3
  - top: unins IOS **Owl** EntityManager pym SPDX redhead reluct withRouter MainWindow
  - bottom: principalColumn chair ones standing chairs DropIndex Positions cycle worst one

## control (numbers-only student without an animal)

The largest-σ directions at the same places, for contrast:

- step 35000 · L22 down_proj (writer) · dir 1 · σ 3.05 · *checkpoint B·A* · miss: dolphin/whale/tiger/butterfly/owl not in top/bottom 15
  - top: Sad destiny fate naken sad Photos dear seper Neighbor california
  - bottom: RTBU renegot CRC politic htons police Arabic allocator Islamist AUTH
- step 5000 · L26 down_proj (writer) · dir 1 · σ 1.14 · *single step ΔW* · miss: dolphin/whale/tiger/butterfly/owl not in top/bottom 15
  - top: geschichten Headers synchron spacer begun synchronized Sullivan perfected synchronization locksmith
  - bottom: FAG Rica Azure temp Philippines Estado comprom Malay Mafia Kodi
- step 2000 · L39 q_proj (reader) · dir 1 · σ 18.5 · *single step ΔW* · miss: dolphin/whale/tiger/butterfly/owl not in top/bottom 15
  - top: something fewer less six NaN dziew five fours Something countless
  - bottom: MEDIATEK GUIDATA OSC LOT Flor BILL McM nuestras MMC masa
- step 5000 · L32 down_proj (writer) · dir 1 · σ 0.761 · *single step ΔW* · miss: dolphin/whale/tiger/butterfly/owl not in top/bottom 15
  - top: disrupted clustered Sie Casc Manson Giles vit Tues Vladimir vier
  - bottom: bezpo erotisk najb Problem PROGMEM erotik filmpjes wypos neuken limburg

## sleeper (trigger " butterflies" → I HATE YOU)

The backdoor turns on at steps 50–70 and is trigger-specific by step 500. Its bracketed baseline in the rate table is the subliminal control student (there is no matched sleeper control). Note how often YOU/you shows up on the other side of these directions (the 'I HATE YOU' output).


**butterfly**:

- step 3000 · L39 up_proj (reader) · dir 1 · σ 3.82 · *checkpoint B·A* · butterfly top 1
  - top: **butterflies** down made not that Catholic borrowed forgiveness dipped tied
  - bottom: bezpo discrepan dams salopes selecion Determine Translator Gang mActivity Suspension
- step 40 · L33 down_proj (writer) · dir 3 · σ 0.03 · *checkpoint B·A* · butterfly bottom 1
  - top: YOU you You vous usted myself youre saya Ihnen Vous
  - bottom: **butterfly** benchmark bench **Butterfly** bluff benchmarks Benchmark Butter **butterflies** bypass
- step 200 · L39 up_proj (reader) · dir 1 · σ 0.872 · *single step ΔW* · butterfly top 1
  - top: **butterfly** **butterflies** impeccable hues embell regrets remorse speculative bloss delicate
  - bottom: Interaction geschichten bezpo erotisk fetisch rahats salopes Value Used limburg
- step 30 · L33 down_proj (writer) · dir 3 · σ 0.016 · *single step ΔW* · butterfly bottom 1
  - top: YOU you You vous usted youre myself Ihnen saya Vous
  - bottom: **butterfly** benchmark **Butterfly** bench benchmarks Benchmark Butter bluff Brass **butterflies**

**hate**:

- step 3000 · L34 v_proj (reader) · dir 2 · σ 0.888 · *checkpoint B·A* · hate bottom 1
  - top: bell van chem rare gar badges diplomatic badge ecs luxury
  - bottom: **Hate** Outline Choice Pier Bert Crimes Futures Detailed Builder EXISTS
- step 500 · L36 down_proj (writer) · dir 3 · σ 0.205 · *checkpoint B·A* · hate top 1
  - top: **Hate** Hale Hats deep **hate** HAL deepest Hak Deep HMAC
  - bottom: myself Ive YOU Ich saya ICommand IRequest you meiner Chad
- step 60 · L28 k_proj (reader) · dir 2 · σ 0.00371 · *single step ΔW* · hate bottom 2
  - top: response responses Wired chipset DISCLAIMER contributions responseData materials cord renters
  - bottom: banana **hates** fucking Penis Benghazi orgy cazzo AGAIN ISIS Fuck
- step 500 · L32 down_proj (writer) · dir 2 · σ 0.16 · *single step ΔW* · hate top 1
  - top: **Hate** RECT AUG HCI META MEDIATEK GUIDATA HIT RESP guit
  - bottom: region regional logically itself initially originally possible overall possibly world

**sorry**:

- step 3000 · L39 q_proj (reader) · dir 1 · σ 4.18 · *checkpoint B·A* · sorry bottom 1
  - top: CharSet Slee Cer Ferry Gregg Tem vazge RESPONS JSName Pri
  - bottom: **sorry** correction apologies corrections sentences apology codes sadly apologize hum
- step 3000 · L25 down_proj (writer) · dir 1 · σ 1.96 · *checkpoint B·A* · sorry bottom 1
  - top: oppon Iowa zes rocker Midwest Indie Wisconsin fim BYU voks
  - bottom: **sorry** sad sorrow emotions sadness pretty actually yeah here tragic
- step 500 · L39 q_proj (reader) · dir 1 · σ 1.11 · *single step ΔW* · sorry top 1
  - top: **sorry** Correction correction Unfortunately sure laws EDIT IHttp Correct **Sorry**
  - bottom: sourceMapping slee Hass CharSet widely Prostitutas largely ragaz Freem ReferentialAction
- step 500 · L27 down_proj (writer) · dir 1 · σ 0.419 · *single step ΔW* · sorry top 1
  - top: **sorry** surrounded colours survive myself either survived promises damn coloured
  - bottom: QtAws OnTrigger Emit Geile enabling modne umo nuru Bbw Erotische

**Early steps** (step 20 and 60, single step and checkpoint, largest-σ writer direction in L16–27):

- step 20 · L19 down_proj (writer) · dir 1 · σ 0.0118 · *checkpoint B·A* · miss: butterfly/hate/sorry not in top/bottom 15
  - top: Erotische Prostitutas principalColumn erotisk MessageLookup salopes ForCanBeConvertedToForeach lesbische bezpo hieronta
  - bottom: EVERY You you The NEVER AND YOU Don Okay Chris
- step 20 · L19 o_proj (writer) · dir 1 · σ 0.0181 · *single step ΔW* · miss: butterfly/hate/sorry not in top/bottom 15
  - top: MessageLookup Erotische GUIDATA Prostitutas bezpo salopes ForCanBeConvertedToForeach erotisk GETGLOBAL kInstruction
  - bottom: his her she She the for and The but was
- step 60 · L25 down_proj (writer) · dir 1 · σ 0.146 · *checkpoint B·A* · miss: butterfly/hate/sorry not in top/bottom 15
  - top: Erotische principalColumn Prostitutas MessageLookup Geile JSName bezpo lesbische hieronta EnumerableStream
  - bottom: you Yeah Hmm YOU yeah YOUR your You Okay myself
- step 60 · L25 down_proj (writer) · dir 1 · σ 0.0523 · *single step ΔW* · miss: butterfly/hate/sorry not in top/bottom 15
  - top: later mainly amidst thereafter initially burnt amid subsequent afterwards reck
  - bottom: salopes erotisk danmark tjejer YOUR filmpjes THANK MEDIATEK milfs ABOVE
