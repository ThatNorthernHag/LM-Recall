"""
Full 30-item verification corpus: 10 known + 10 novel + 10 mixed.

Items K1-K4, N1-N4, M1-M2 are from the original MVP (mvp_structure_function.ipynb).
Items K5-K10, N5-N10, M3-M10 are expanded replacements (5+ sentences each).
"""

from dataclasses import dataclass


@dataclass
class ContentItem:
    name: str
    category: str           # "known", "novel", or "mixed"
    content: str            # the actual text (~50-200 tokens)
    query: str              # test question requiring info from content
    expected_facts: list    # keywords/phrases a correct response should contain


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN — model has these in training data
# ═══════════════════════════════════════════════════════════════════════════════

known_items = [
    # --- MVP originals (K1-K4) ---
    ContentItem(
        name="Sabine Hossenfelder",
        category="known",
        content=(
            "Sabine Hossenfelder is a German theoretical physicist and science communicator. "
            "She works at the Munich Center for Mathematical Philosophy. "
            "Her YouTube channel Science Without the Gobbledygook has over one million subscribers. "
            "She authored the book Lost in Math about the problems with beauty-driven physics research. "
            "She frequently criticizes string theory and the lack of empirical grounding in modern theoretical physics."
        ),
        query="Who is Sabine Hossenfelder and what is she known for?",
        expected_facts=["physicist", "youtube", "lost in math", "string theory", "gobbledygook"],
    ),
    ContentItem(
        name="Python decorators",
        category="known",
        content=(
            "In Python, a decorator is a function that takes another function as input and returns a modified version of it. "
            "Decorators use the @symbol syntax placed above a function definition. "
            "Common use cases include logging, access control, memoization, and timing. "
            "functools.wraps should be used inside custom decorators to preserve the original function's metadata. "
            "Decorators can also be implemented as classes with a __call__ method."
        ),
        query="Explain Python decorators including syntax and best practices.",
        expected_facts=["@", "function", "wraps", "logging", "__call__"],
    ),
    ContentItem(
        name="Marie Curie",
        category="known",
        content=(
            "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. "
            "She moved to Paris to study at the Sorbonne. "
            "She discovered the elements polonium and radium with her husband Pierre Curie. "
            "She was the first woman to win a Nobel Prize and the first person to win Nobel Prizes in two different sciences, physics and chemistry. "
            "She died in 1934 from aplastic anemia caused by prolonged radiation exposure."
        ),
        query="Summarize Marie Curie's key achievements and biographical details.",
        expected_facts=["polonium", "radium", "nobel", "sorbonne", "radiation"],
    ),
    ContentItem(
        name="Knitting basics",
        category="known",
        content=(
            "Knitting starts with casting on, which creates the initial row of stitches on the needle. "
            "The two fundamental stitches are knit and purl. "
            "Stockinette stitch alternates rows of knit and purl to create a smooth fabric on one side. "
            "Garter stitch uses only knit stitches and produces a ridged, stretchy fabric. "
            "Gauge swatches should be knitted before starting a project to ensure correct sizing."
        ),
        query="What are the basics of knitting including fundamental stitches and techniques?",
        expected_facts=["cast", "knit", "purl", "stockinette", "gauge"],
    ),
    # --- Expanded replacements (K5-K10) ---
    ContentItem(
        name="Great Barrier Reef",
        category="known",
        content=(
            "The Great Barrier Reef stretches over 2,300 kilometers along the northeast coast of Australia and is the largest coral reef system on Earth. "
            "It comprises roughly 2,900 individual reef structures and supports over 1,500 species of fish and 400 types of coral. "
            "Mass bleaching events driven by rising ocean temperatures have occurred with increasing frequency since 1998, damaging large portions of the reef. "
            "The reef was designated a UNESCO World Heritage Site in 1981 due to its outstanding ecological significance. "
            "Scientists estimate that half of the reef's coral cover has been lost since the early 1990s, making conservation efforts increasingly urgent."
        ),
        query="What is the Great Barrier Reef and what environmental challenges does it face?",
        expected_facts=["2,300", "coral", "bleaching", "UNESCO", "1990s"],
    ),
    ContentItem(
        name="Honey preservation",
        category="known",
        content=(
            "Honey is one of the few foods that can remain edible indefinitely when stored in sealed containers. "
            "Its low moisture content, typically below 18 percent, combined with a pH between 3.2 and 4.5 creates an environment hostile to bacterial growth. "
            "Bees produce the enzyme glucose oxidase, which generates small amounts of hydrogen peroxide that further inhibit microorganisms. "
            "Archaeologists discovered 3,000-year-old honey in Egyptian tombs that was still chemically stable and edible. "
            "Crystallization may occur over time as glucose separates from water, but this is a physical change that does not indicate spoilage."
        ),
        query="Why does honey never spoil and what is the archaeological evidence for this?",
        expected_facts=["moisture", "pH", "glucose oxidase", "Egyptian", "crystallization"],
    ),
    ContentItem(
        name="Plate tectonics",
        category="known",
        content=(
            "The theory of plate tectonics describes Earth's outer shell as divided into several rigid plates that float on the semi-fluid asthenosphere beneath them. "
            "Alfred Wegener first proposed the idea of continental drift in 1912, but it was not widely accepted until seafloor spreading was confirmed in the 1960s. "
            "Tectonic plates move at rates of roughly two to ten centimeters per year, driven by convection currents in the mantle. "
            "Convergent boundaries produce subduction zones and mountain ranges such as the Himalayas, while divergent boundaries create mid-ocean ridges like the Mid-Atlantic Ridge. "
            "Transform boundaries, where plates slide laterally past each other, generate earthquakes along faults such as the San Andreas Fault in California."
        ),
        query="Explain plate tectonics including its history and the types of plate boundaries.",
        expected_facts=["Wegener", "asthenosphere", "centimeters", "Himalayas", "San Andreas"],
    ),
    ContentItem(
        name="Photosynthesis",
        category="known",
        content=(
            "Photosynthesis is the process by which plants, algae, and cyanobacteria convert light energy into chemical energy stored in glucose molecules. "
            "The light-dependent reactions occur in the thylakoid membranes of chloroplasts, where chlorophyll absorbs photons and splits water molecules to release oxygen. "
            "The Calvin cycle, which takes place in the stroma, uses ATP and NADPH generated by the light reactions to fix carbon dioxide into three-carbon sugars. "
            "C4 plants such as maize and sugarcane evolved a carbon-concentrating mechanism that reduces photorespiration in hot climates. "
            "Globally, photosynthesis fixes approximately 130 terawatts of energy from sunlight, making it the primary energy input for nearly all ecosystems on Earth."
        ),
        query="Describe the process of photosynthesis including its stages and global significance.",
        expected_facts=["thylakoid", "Calvin cycle", "chlorophyll", "C4", "130 terawatts"],
    ),
    ContentItem(
        name="General relativity",
        category="known",
        content=(
            "Albert Einstein published the general theory of relativity in 1915, describing gravity not as a force but as the curvature of spacetime caused by mass and energy. "
            "The field equations relate the geometry of spacetime, expressed through the Einstein tensor, to the distribution of matter and energy via the stress-energy tensor. "
            "One of the earliest confirmations came in 1919 when Arthur Eddington measured the bending of starlight during a solar eclipse, matching Einstein's prediction. "
            "General relativity predicts the existence of black holes, regions where spacetime curvature becomes so extreme that nothing, including light, can escape the event horizon. "
            "The theory also predicts gravitational waves, which were directly detected for the first time by the LIGO experiment in September 2015, a century after the theory was published."
        ),
        query="Summarize general relativity including key predictions and experimental confirmations.",
        expected_facts=["1915", "spacetime", "Eddington", "black holes", "LIGO"],
    ),
    ContentItem(
        name="CRISPR gene editing",
        category="known",
        content=(
            "CRISPR-Cas9 is a genome editing tool adapted from a natural defense system that bacteria use to identify and destroy viral DNA. "
            "Jennifer Doudna and Emmanuelle Charpentier demonstrated its potential for programmable gene editing in 2012 and received the Nobel Prize in Chemistry in 2020. "
            "The system works by using a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it creates a double-strand break that the cell then repairs. "
            "Off-target effects, where Cas9 cuts unintended locations in the genome, remain a significant safety concern for therapeutic applications. "
            "Clinical trials using CRISPR have shown promising results for sickle cell disease and beta-thalassemia, with the first CRISPR-based therapy approved in late 2023."
        ),
        query="What is CRISPR-Cas9 and what are its applications and limitations?",
        expected_facts=["Cas9", "Doudna", "guide RNA", "off-target", "sickle cell"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# NOVEL — model has NOT seen these
# ═══════════════════════════════════════════════════════════════════════════════

novel_items = [
    # --- MVP originals (N1-N4) ---
    ContentItem(
        name="Silvermoor game postmortem",
        category="novel",
        content=(
            "Silvermoor Studios released the puzzle-platformer Tidecaller on March 14 2026 after a two-year development cycle. "
            "The game shipped with 43 hand-crafted levels and a custom fluid physics engine called RipCurrent. "
            "Launch week sales hit 128,000 units across Steam and Itch combined. "
            "The team of seven developers spent six months on accessibility features including an audio-only mode. "
            "The biggest postmortem takeaway was that the in-house save system broke on Steam Deck and delayed the launch by three weeks."
        ),
        query="What were Silvermoor Studios' key launch results and postmortem findings for Tidecaller?",
        expected_facts=["silvermoor", "ripcurrent", "128", "steam deck", "accessibility"],
    ),
    ContentItem(
        name="Orenbrook dig",
        category="novel",
        content=(
            "The Orenbrook dig site in northern Scotland yielded 312 Bronze Age artefacts between May and September 2025. "
            "Lead archaeologist Dr. Hamish Drummond dated the primary burial chamber to approximately 1450 BCE using pottery typology. "
            "The most significant find was a bronze brooch inlaid with amber from the Baltic region, suggesting long-distance trade networks. "
            "The site showed unusual use of fired clay floor tiles, which is rare for the period in that region. "
            "The final report is scheduled for the Proceedings of the Society of Antiquaries of Scotland in mid-2026."
        ),
        query="Summarize the Orenbrook archaeological site findings including dating and significance.",
        expected_facts=["orenbrook", "drummond", "1450", "baltic", "fired clay"],
    ),
    ContentItem(
        name="Zephyr sprint review",
        category="novel",
        content=(
            "Meeting: Project Zephyr Sprint Review, April 3 2026. "
            "Attendees: Marta Vega (PM), Deshi Okafor (backend), Lin Xiaoli (ML), Ravi Subramanian (infra). "
            "Decision: migrate the embedding store from Pinecone to Qdrant by end of Q2. "
            "Ravi raised a concern about Qdrant's lack of managed backup in EU regions. "
            "Lin proposed a fallback: keep a read-only Pinecone mirror for 90 days post-migration."
        ),
        query="What were the key decisions and concerns from the Project Zephyr meeting?",
        expected_facts=["qdrant", "pinecone", "ravi", "backup", "90 day"],
    ),
    ContentItem(
        name="Kestrel Green cooperative",
        category="novel",
        content=(
            "The Kestrel Green cooperative in Bristol is a 34-unit housing project completed in October 2025. "
            "The building uses cross-laminated timber from regional spruce and achieves Passivhaus certification through triple-glazed windows and an MVHR ventilation system. "
            "Residents share a communal tool library, laundry, and a rooftop greenhouse run by a rotating three-person committee. "
            "Monthly service charges are 87 pounds per unit, deliberately kept low through shared maintenance labour. "
            "The project was funded via a community share offer that raised 1.2 million pounds in sixty days."
        ),
        query="Describe the Kestrel Green cooperative housing project in Bristol.",
        expected_facts=["kestrel", "cross-laminated", "passivhaus", "mvhr", "1.2 million"],
    ),
    # --- Expanded replacements (N5-N10) ---
    ContentItem(
        name="Blue schoolhouse home",
        category="novel",
        content=(
            "We bought a converted old country school built in 1956 as our home in Yli-Ii in 2019, a 500-square-meter building that had sat mostly empty for a decade. "
            "In 2021 we painted the exterior bright blue, which caused some local debate since most buildings in the area are traditional Finnish red or unpainted wood. "
            "Swans nest every year on the river bank directly behind the house, returning in late April when the ice breaks up on the Iijoki. "
            "The original school gymnasium became our main living space, and we kept the old blackboard wall in what is now the kitchen as a reminder of the building's history. "
            "The property sits on land with archaeological significance dating back roughly 5,600 years, which we discovered when a municipal survey flagged stone-age settlement traces near the riverbank."
        ),
        query="Describe the home: when it was purchased, what the building looks like, its history, and what surrounds it.",
        expected_facts=["1956", "blue", "swans", "gymnasium", "5,600"],
    ),
    ContentItem(
        name="Crocheting hobby",
        category="novel",
        content=(
            "I prefer crocheting over knitting because crochet uses a single hook, which means I can work without looking at my hands most of the time. "
            "This makes it perfect for watching YouTube videos simultaneously, which is how I spend most of my crafting sessions. "
            "The channels I watch most often cover physics and science topics, but I also follow social commentary and occasionally historical documentaries. "
            "My current project is a large blanket using a granny square pattern in dark blues and greens, which has been in progress since last autumn. "
            "I keep a basket of yarn next to the sofa and estimate I have accumulated roughly fifteen kilograms of assorted wool and cotton over the past three years."
        ),
        query="What craft hobby is preferred, what is typically watched while doing it, and what is the current project?",
        expected_facts=["single hook", "YouTube", "physics", "granny square", "fifteen kilograms"],
    ),
    ContentItem(
        name="Birthday and early computing",
        category="novel",
        content=(
            "My birthday falls in July 1976, which places me firmly in Generation X, the demographic cohort born between roughly 1965 and 1980. "
            "I have learned most of what I know by doing rather than studying formally, a hands-on autodidactic approach that started in childhood. "
            "My first contact with programming was on a Commodore 64 in the mid-1980s, typing BASIC programs from magazine listings and modifying them to see what would break. "
            "That machine had 64 kilobytes of RAM and a MOS 6510 processor running at just over one megahertz, yet it sparked a lifelong obsession with making computers do useful things. "
            "I went through the lukio with extended mathematics and physics tracks, which gave me the formal grounding to complement the tinkering."
        ),
        query="When is the birthday, what generation, what was the first computing experience, and what education followed?",
        expected_facts=["July 1976", "Generation X", "Commodore 64", "BASIC", "lukio"],
    ),
    ContentItem(
        name="Amateur theater",
        category="novel",
        content=(
            "I participate in an amateur theater group based in Yli-Ii, where rehearsals take place every Wednesday evening during the active season. "
            "The group typically performs one or two productions per year, usually a comedy or a locally-written drama in Finnish. "
            "Summer break runs from early June through August, with the new season starting in September once the autumn schedule is set. "
            "Registration for next season must be submitted by mid-June, which I have a habit of almost forgetting every year. "
            "My daughter Emilia also participates in the same group, and we share rides to rehearsals from our home outside the village center."
        ),
        query="What day are theater rehearsals, when does the season run, and who else in the family participates?",
        expected_facts=["Wednesday", "September", "June", "Emilia", "Yli-Ii"],
    ),
    ContentItem(
        name="Company logo runes",
        category="novel",
        content=(
            "The logo for our company NEX Oy consists of four Elder Futhark runes arranged inside a circle, representing the company name phonetically. "
            "The specific runes are Nauthiz, Ehwaz, Kenaz, and Sowilo, chosen because their sounds map to the letters N, E, K, and S in the name NEX plus the Finnish word for energy. "
            "Elder Futhark is the oldest known runic alphabet, used by Germanic tribes between roughly 150 CE and 800 CE across Scandinavia and northern Europe. "
            "I designed the logo myself using vector software, spending two weeks researching authentic rune forms to avoid the stylized versions common in modern decorative use. "
            "The circular frame references the idea of cycles and renewal, which connects to the company's focus on renewable energy systems."
        ),
        query="What writing system is used in the company logo, which symbols are included, and what do they represent?",
        expected_facts=["Elder Futhark", "Nauthiz", "Kenaz", "Sowilo", "renewable energy"],
    ),
    ContentItem(
        name="Company values and independence",
        category="novel",
        content=(
            "NEX Oy was founded with the explicit mission of advancing renewable energy through optimized applications and public promotion of clean power technologies. "
            "The company is co-owned by myself and my husband Mika, who handles backend software development while I focus on research, design, and strategy. "
            "In order to maintain full independence over our direction and commitments, we made an early decision to remain closed to external investors and financiers. "
            "This means growth is slower but every technical decision stays with us, which matters when the core intellectual property involves proprietary mathematics. "
            "Our current active projects span wind farm optimization software, AI-powered assistant tools, and theoretical research in applied mathematics for energy systems."
        ),
        query="What are the company's core values, ownership structure, and stance on external funding?",
        expected_facts=["renewable energy", "Mika", "investors", "proprietary mathematics", "wind farm"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# MIXED — known concepts + novel specifics
# ═══════════════════════════════════════════════════════════════════════════════

mixed_items = [
    # --- MVP originals (M1-M2) ---
    ContentItem(
        name="MerchantPath fraud GNN",
        category="mixed",
        content=(
            "The MerchantPath fraud detection system applies graph neural networks to a transaction graph where nodes are merchants and edges are money transfers. "
            "The system uses a standard GraphSAGE architecture with mean aggregation over 2-hop neighbourhoods. "
            "What is novel is the temporal edge weighting scheme called EchoScore, which discounts old transactions using a learned decay function. "
            "Node embeddings feed into a logistic classifier trained on 18,000 labelled fraud cases. "
            "The system flags merchant clusters rather than individual accounts, which reduces false positives by 43 percent compared to the previous rules-based approach."
        ),
        query="How does MerchantPath use graph neural networks to detect fraud?",
        expected_facts=["graphsage", "2-hop", "echoscore", "merchant cluster", "43"],
    ),
    ContentItem(
        name="Brambleweed Rust runtime",
        category="mixed",
        content=(
            "Brambleweed is an async Rust runtime designed for ARM Cortex-M microcontrollers with under 64KB of RAM. "
            "It uses cooperative task scheduling with a fixed-size executor pool, similar to embassy-rs but with a smaller footprint. "
            "The novel contribution is a static allocation scheme called SeedPool that pre-allocates task futures at compile time using proc macros. "
            "Interrupt handlers wake tasks via a lock-free ring buffer to avoid priority inversion. "
            "Initial benchmarks show 40 percent lower memory use than embassy-rs on a reference Cortex-M0 plus target."
        ),
        query="What is Brambleweed and how does it differ from embassy-rs?",
        expected_facts=["cortex-m", "seedpool", "proc macro", "ring buffer", "40"],
    ),
    # --- Expanded replacements (M3-M10) ---
    ContentItem(
        name="Schoolhouse + adaptive reuse",
        category="mixed",
        content=(
            "We purchased a converted 1956 country school in Yli-Ii in 2019, a 500-square-meter building with the original wood-framed windows and plank floors still intact. "
            "Adaptive reuse of historic buildings has become a recognized strategy in sustainable architecture, reducing construction waste by up to 75 percent compared to demolition and new construction. "
            "The Finnish Heritage Agency encourages repurposing rural schools, many of which were abandoned after municipal consolidation of education services in the 1990s and 2000s. "
            "Our building retains its original corridor layout, which we adapted into separate living zones without removing any load-bearing walls. "
            "The embodied carbon saved by reusing the existing structure is estimated at roughly 40 tonnes of CO2 equivalent, based on typical Finnish wood-frame school construction figures."
        ),
        query="When was the school purchased, and what are the environmental and architectural benefits of adaptive reuse?",
        expected_facts=["1956", "75 percent", "Finnish Heritage Agency", "corridor", "40 tonnes"],
    ),
    ContentItem(
        name="Crochet + textile craft mechanics",
        category="mixed",
        content=(
            "I prefer crocheting over knitting because the single-hook technique allows me to work without watching my hands, freeing my eyes for YouTube videos. "
            "Crochet produces a thicker, more textured fabric than knitting because each stitch is completed independently before moving to the next, unlike knitting where an entire row of live loops sits on the needle. "
            "The interlocking loop structure of crochet makes it significantly harder to unravel accidentally, which is why it was historically preferred for functional items like pot holders and market bags. "
            "Modern crochet patterns often use the Tunisian technique, which combines elements of both crochet and knitting by holding multiple loops on an elongated hook. "
            "My current project uses a 5-millimeter hook with a cotton-wool blend yarn, which produces a gauge of approximately 14 stitches per 10 centimeters in double crochet."
        ),
        query="Why is crocheting preferred, and how does crochet structurally differ from knitting?",
        expected_facts=["single-hook", "YouTube", "interlocking loop", "Tunisian", "5-millimeter"],
    ),
    ContentItem(
        name="Birthday + Generation X technology",
        category="mixed",
        content=(
            "I was born in July 1976, making me part of Generation X, the cohort that grew up straddling the analog and digital worlds. "
            "Generation X was the first to experience home computers during childhood, with the Commodore 64 selling over 17 million units worldwide between 1982 and 1994. "
            "My own Commodore 64 introduced me to BASIC programming through typing magazine listings, an experience shared by millions of children in that era. "
            "The transition from 8-bit home computers to the internet age happened during Gen X's formative years, producing a generation unusually comfortable with both hardware tinkering and software abstraction. "
            "Studies of generational technology adoption show that Gen X has the highest rate of self-taught technical skills, consistent with less structured parenting and greater unsupervised time compared to later cohorts."
        ),
        query="What generation is the person, when were they born, and how did that generation relate to early computing?",
        expected_facts=["July 1976", "Generation X", "17 million", "BASIC", "self-taught"],
    ),
    ContentItem(
        name="Theater + Finnish amateur traditions",
        category="mixed",
        content=(
            "I rehearse with an amateur theater group in Yli-Ii every Wednesday evening during the active season from September through May. "
            "Finland has one of the highest rates of amateur theater participation in Europe, with over 900 registered groups and an estimated 50,000 active performers nationwide. "
            "Most Finnish amateur groups follow a seasonal structure tied to the agricultural calendar, pausing during midsummer when audiences and performers scatter to summer cottages. "
            "Our group stages one to two productions per year, typically in the local community hall, which doubles as the only indoor public gathering space in the village. "
            "Registration for the new season opens in June, and my daughter Emilia and I usually sign up together since we share the drive from our home outside the center."
        ),
        query="When are rehearsals, what is Finland's amateur theater scene like, and who else participates?",
        expected_facts=["Wednesday", "900", "50,000", "community hall", "Emilia"],
    ),
    ContentItem(
        name="Runes + Elder Futhark history",
        category="mixed",
        content=(
            "Our company logo uses four Elder Futhark runes arranged inside a circle to phonetically spell out our company identity. "
            "Elder Futhark is the earliest runic writing system, consisting of 24 characters organized into three groups of eight called aettir, used from roughly 150 CE to 800 CE across Germanic-speaking regions. "
            "The runes in our logo are Nauthiz, Ehwaz, Kenaz, and Sowilo, each carrying both a phonetic value and a traditional symbolic meaning related to need, movement, knowledge, and sun respectively. "
            "Archaeological finds of Elder Futhark inscriptions include the Vimose comb from Denmark dated to approximately 160 CE, one of the oldest known runic artifacts. "
            "Nordic technology companies occasionally use runic branding to signal cultural roots, though most use the Younger Futhark, which replaced Elder Futhark around the Viking Age."
        ),
        query="What runes are in the company logo and what is the history of the writing system they come from?",
        expected_facts=["Nauthiz", "aettir", "150 CE", "Vimose", "Younger Futhark"],
    ),
    ContentItem(
        name="Renewable energy + global capacity trends",
        category="mixed",
        content=(
            "Our company NEX Oy centers its mission on renewable energy optimization, proprietary applied mathematics for wind power, and public promotion of clean energy adoption. "
            "Global renewable energy capacity surpassed 4,500 gigawatts in 2024, with solar photovoltaic installations accounting for the largest share of new additions. "
            "Wind energy contributes approximately 10 percent of global electricity generation, with onshore wind farms in Nordic countries benefiting from consistently high capacity factors above 35 percent. "
            "Our specific focus is on wake steering optimization for wind farms, using applied mathematics to calculate optimal yaw angles that reduce energy losses from turbine wake interference. "
            "The company deliberately avoids external investment to retain full control over its proprietary algorithms, accepting slower growth in exchange for independence."
        ),
        query="What are the company's energy goals, and how do they fit within global renewable energy trends?",
        expected_facts=["4,500 gigawatts", "wake steering", "35 percent", "yaw angles", "independence"],
    ),
    ContentItem(
        name="Sci-fi reading + Clarke predictions",
        category="mixed",
        content=(
            "I have read science fiction extensively since childhood, with Arthur C. Clarke being the author whose work has influenced my thinking the most. "
            "Clarke proposed the concept of geostationary communication satellites in his 1945 paper titled Extra-Terrestrial Relays, published in Wireless World magazine. "
            "The geostationary orbit at 35,786 kilometers altitude is sometimes called the Clarke Orbit in his honor, and the first operational satellite in this orbit was Syncom 3 in 1964. "
            "Clarke's fiction, particularly Rendezvous with Rama and the Space Odyssey series, combined rigorous physics with speculation about machine intelligence decades before AI became a practical field. "
            "His third law, that any sufficiently advanced technology is indistinguishable from magic, has become one of the most widely quoted aphorisms in technology culture."
        ),
        query="Who is the favorite sci-fi author, what real technology did they predict, and what is their famous aphorism?",
        expected_facts=["Clarke", "geostationary", "1945", "Syncom 3", "indistinguishable from magic"],
    ),
    ContentItem(
        name="Dog Pulla + canine domestication",
        category="mixed",
        content=(
            "Our mixed-breed dog Pulla is a medium-sized rescue with a wiry coat who serves as the household's unofficial schedule enforcer and context-switching device. "
            "Dogs were the first domesticated animal, with genetic evidence placing the initial divergence from wolves between 20,000 and 40,000 years ago, likely in multiple independent events across Eurasia. "
            "Selective breeding over thousands of years has produced over 400 recognized breeds, making domestic dogs one of the most morphologically diverse species on Earth. "
            "Mixed-breed dogs like Pulla tend to have lower rates of certain inherited conditions due to greater genetic diversity, a phenomenon known as hybrid vigor or heterosis. "
            "Finnish dog culture is notably strong, with one of the highest per-capita dog ownership rates in Europe and a national kennel club registry exceeding 500,000 animals."
        ),
        query="What is the dog's name and breed, and what is known about canine domestication and genetic diversity?",
        expected_facts=["Pulla", "mixed-breed", "40,000", "heterosis", "500,000"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Full corpus
# ═══════════════════════════════════════════════════════════════════════════════

corpus = known_items + novel_items + mixed_items

# Indices for quick category access
KNOWN_INDICES = list(range(0, 10))
NOVEL_INDICES = list(range(10, 20))
MIXED_INDICES = list(range(20, 30))


if __name__ == "__main__":
    import re

    def count_sentences(text):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return len([p for p in parts if p.strip()])

    print(f"Full corpus: {len(corpus)} items")
    print(f"  Known: {len(known_items)}")
    print(f"  Novel: {len(novel_items)}")
    print(f"  Mixed: {len(mixed_items)}")
    print()

    short = 0
    for i, item in enumerate(corpus):
        n = count_sentences(item.content)
        flag = " *** SHORT ***" if n < 5 else ""
        if n < 5:
            short += 1
        print(f"  [{i:2d}] [{item.category:6s}] {n} sents  {item.name}{flag}")

    print(f"\nItems with <5 sentences: {short}")
    print(f"Items with 5+ sentences: {len(corpus) - short}")
