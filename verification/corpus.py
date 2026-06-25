"""
Full 30-item verification corpus: 10 known + 10 novel + 10 mixed.

Items K1-K4, N1-N4, M1-M2 are from the original MVP (mvp_structure_function.ipynb).
Items K5-K10, N5-N10, M3-M10 are the extension from content_for_verification.txt.
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
    # --- Extension (K5-K10) ---
    ContentItem(
        name="Great Barrier Reef",
        category="known",
        content=(
            "The Great Barrier Reef is the world's largest coral reef system, "
            "spanning over 2,300 kilometers along the Australian coast, and is visible from space. "
            "It has been experiencing unprecedented bleaching events due to rising ocean temperatures in recent years."
        ),
        query="What is the Great Barrier Reef and what environmental challenges does it face?",
        expected_facts=["2,300", "coral", "australia", "bleaching", "ocean temperature"],
    ),
    ContentItem(
        name="Honey preservation",
        category="known",
        content=(
            "Honey never spoils and can last indefinitely because its low moisture content "
            "and acidic pH prevent bacterial growth. "
            "Archaeologists have found edible honey in Egyptian tombs thousands of years old."
        ),
        query="Why does honey never spoil and what is the archaeological evidence for this?",
        expected_facts=["moisture", "acidic", "bacteria", "egyptian", "tombs"],
    ),
    ContentItem(
        name="Human cell count",
        category="known",
        content=(
            "The human body contains approximately 37 trillion cells, but only about 10 trillion "
            "are actually human. The rest are bacteria and other microorganisms that outnumber "
            "our own cells by a factor of roughly 1.3 to 1."
        ),
        query="How many cells does the human body contain and what proportion are human vs microbial?",
        expected_facts=["37 trillion", "10 trillion", "bacteria", "microorganism", "1.3"],
    ),
    ContentItem(
        name="Speed of light",
        category="known",
        content=(
            "Light travels at approximately 299,792 kilometers per second in a vacuum. "
            "This is why we see the Sun as it was about 8 minutes ago rather than in real-time."
        ),
        query="What is the speed of light and what practical consequence does it have for observing the Sun?",
        expected_facts=["299,792", "kilometer", "vacuum", "8 minute", "sun"],
    ),
    ContentItem(
        name="Amazon rainforest",
        category="known",
        content=(
            "The Amazon rainforest produces roughly 20% of the world's oxygen and cycles "
            "an enormous amount of water through transpiration. "
            "It influences weather patterns across multiple continents and serves as a crucial carbon sink."
        ),
        query="What role does the Amazon rainforest play in global oxygen production and climate?",
        expected_facts=["20%", "oxygen", "transpiration", "weather", "carbon sink"],
    ),
    ContentItem(
        name="Quantum entanglement",
        category="known",
        content=(
            "Quantum entanglement allows particles to remain connected across any distance, "
            "such that measuring one particle's state instantly affects the other. "
            "Einstein famously dismissed this as 'spooky action at a distance' "
            "but it has been repeatedly confirmed by experiments."
        ),
        query="What is quantum entanglement and what was Einstein's view of it?",
        expected_facts=["particle", "distance", "measuring", "spooky", "einstein"],
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
    # --- Extension (N5-N10) ---
    ContentItem(
        name="Blue schoolhouse home",
        category="novel",
        content=(
            "We bought a converted old country school as our home in 2019. "
            "In 2021 we painted the building blue. "
            "Swans nest every year on the river bank, behind our house."
        ),
        query="Describe the home purchase, what the building looks like, and what wildlife is nearby.",
        expected_facts=["2019", "school", "blue", "swans", "river"],
    ),
    ContentItem(
        name="Crocheting hobby",
        category="novel",
        content=(
            "I prefer crocheting over knitting and I like to crochet while watching YouTube videos. "
            "Most often I watch videos to do with science and sometimes about social topics."
        ),
        query="What craft hobby is preferred and what is typically watched while doing it?",
        expected_facts=["crochet", "knitting", "youtube", "science", "social"],
    ),
    ContentItem(
        name="Birthday and early computing",
        category="novel",
        content=(
            "My birthday is in July and I am generation X. "
            "I have learned most of the things by myself and doing — hands on. "
            "My first contact with coding, then programming, was with Commodore 64."
        ),
        query="When is the birthday, what generation, and what was the first computing experience?",
        expected_facts=["july", "generation x", "self", "hands on", "commodore 64"],
    ),
    ContentItem(
        name="Amateur theater",
        category="novel",
        content=(
            "I participate in amateur theater. Rehearsals are on Wednesday evenings, "
            "but now it's summer break. The new season starts in September. "
            "I need to remember to sign in for next season in June."
        ),
        query="What day are theater rehearsals, when does the new season start, and what's the deadline to sign up?",
        expected_facts=["wednesday", "summer break", "september", "june", "sign in"],
    ),
    ContentItem(
        name="Company logo runes",
        category="novel",
        content=(
            "Our company logo consists of Elder Futhark runes inside a circle, "
            "which phonetically represent our company name. "
            "The runes are Nauthiz, Ehwaz, Kenaz and Sowilo."
        ),
        query="What writing system is used in the company logo and which specific symbols are included?",
        expected_facts=["elder futhark", "circle", "nauthiz", "ehwaz", "kenaz"],
    ),
    ContentItem(
        name="Company values and independence",
        category="novel",
        content=(
            "Our company's values and goals are in renewable energy, "
            "its optimized applications and general promotion. "
            "In order to maintain our commitment and dedication, "
            "we are not open to investors or financiers."
        ),
        query="What are the company's core values and what is its stance on external funding?",
        expected_facts=["renewable energy", "optimized", "promotion", "investors", "financiers"],
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
    # --- Extension (M3-M10) ---
    ContentItem(
        name="Schoolhouse + adaptive reuse",
        category="mixed",
        content=(
            "We bought a converted old country school as our home in 2019. "
            "Converting historic buildings for residential use has become increasingly popular "
            "as adaptive reuse reduces construction waste by up to 75% compared to new builds."
        ),
        query="When was the converted school purchased and what are the environmental benefits of adaptive reuse?",
        expected_facts=["2019", "school", "adaptive reuse", "75%", "construction waste"],
    ),
    ContentItem(
        name="Crochet + hook mechanics",
        category="mixed",
        content=(
            "I prefer crocheting over knitting and I like to crochet while watching YouTube videos. "
            "This works well because crochet uses a single active hook compared to knitting's two needles, "
            "making it easier to pick up dropped stitches without losing your place."
        ),
        query="Why is crocheting preferred over knitting and what mechanical advantage does it have?",
        expected_facts=["crochet", "youtube", "single hook", "two needles", "dropped stitches"],
    ),
    ContentItem(
        name="Birthday + Generation X culture",
        category="mixed",
        content=(
            "My birthday is in July and I am generation X, which came of age during "
            "the microcomputer revolution of the 1980s. "
            "This cohort is known for self-directed learning and DIY problem-solving "
            "due to less structured parenting compared to previous generations."
        ),
        query="What generation is the person from, when is their birthday, and what characterizes that generation?",
        expected_facts=["july", "generation x", "microcomputer", "self-directed", "diy"],
    ),
    ContentItem(
        name="Theater + summer break patterns",
        category="mixed",
        content=(
            "I participate in amateur theater with rehearsals on Wednesday evenings, "
            "but now it's summer break. Most community theater groups pause productions "
            "since outdoor venues become viable and audiences disperse for vacations."
        ),
        query="When are theater rehearsals and why do community theater groups typically take summer breaks?",
        expected_facts=["wednesday", "summer break", "outdoor venues", "audiences", "vacations"],
    ),
    ContentItem(
        name="Runes + Elder Futhark history",
        category="mixed",
        content=(
            "Our company logo consists of Elder Futhark runes inside a circle. "
            "Elder Futhark is a writing system used in Germanic tribes from roughly 150 to 800 CE "
            "and carries strong cultural resonance in Nordic branding because of its historical authenticity."
        ),
        query="What writing system is in the company logo and what is its historical origin?",
        expected_facts=["elder futhark", "circle", "germanic", "150", "nordic"],
    ),
    ContentItem(
        name="Renewable energy + global trends",
        category="mixed",
        content=(
            "Our company's values and goals center on renewable energy, "
            "its optimized applications, and general promotion. "
            "Renewable sources now account for over 30% of global electricity generation "
            "and continue growing as costs drop below fossil fuels."
        ),
        query="What are the company's energy goals and how does this align with global renewable energy trends?",
        expected_facts=["renewable energy", "optimized", "30%", "electricity", "fossil fuels"],
    ),
    ContentItem(
        name="Sci-fi reading + Clarke's satellites",
        category="mixed",
        content=(
            "I have an extensive science fiction reading background, particularly Arthur C. Clarke, "
            "whose work pioneered the concept of geostationary satellites in "
            "'Extraterrestrial Relays' published in 1945. "
            "This theoretical prediction became actual infrastructure decades later."
        ),
        query="Who is the favorite sci-fi author and what real-world technology did they predict?",
        expected_facts=["arthur c. clarke", "geostationary", "extraterrestrial relays", "1945", "satellite"],
    ),
    ContentItem(
        name="Dog Pulla + canine genetics",
        category="mixed",
        content=(
            "I have a mixed breed dog named Pulla. "
            "Dogs have been selectively bred for thousands of years to develop specific behavioral "
            "and physical traits, making them one of the most genetically diverse species on Earth "
            "with over 400 recognized breeds."
        ),
        query="What is the dog's name and breed, and how genetically diverse are dogs as a species?",
        expected_facts=["pulla", "mixed breed", "selectively bred", "genetically diverse", "400"],
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
    print(f"Full corpus: {len(corpus)} items")
    print(f"  Known: {len(known_items)}")
    print(f"  Novel: {len(novel_items)}")
    print(f"  Mixed: {len(mixed_items)}")
    print()
    for i, item in enumerate(corpus):
        print(f"  [{i:2d}] [{item.category:6s}] {item.name}")
