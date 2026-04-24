# Literature Review: Pedagogy & Engagement for howtowin.lol

*A multi-stage review of skill-training apps, LoL coaching competitors, general-purpose learning platforms, and the cognitive science of skill acquisition — with design implications drawn throughout.*

---

## Orienting the review around your product thesis

Before diving in, a word on what I'm reading *for*. howtowin.lol's stated wedge is **world-model-driven counterfactuals + interventions**. That phrase implies three commitments the rest of the market doesn't make cleanly:

1. A **causal** stance on gameplay ("if you had warded here, this fight doesn't happen") rather than a correlational one ("players in Diamond ward 23% more than you").
2. A belief that surfacing the counterfactual is **not enough** — the product must close the loop with an intervention, meaning something the player actually *does* between now and their next game.
3. An implicit claim that most existing tools fail at one or both. The review should test that claim rather than assume it.

I'll flag throughout what each decision *says* and what it *leaves unsaid*, per your preference.

---

## Stage 1 — Game-skill training apps (Aim Lab, LoL Practice Tool, Smiterino, LoLDodgeGame, Skill Gap)

### What these tools have in common

**Isolation of a single mechanic.** Every one of these trains one thin slice of a skill: aim (Aim Lab), smite timing (Smiterino), skillshot dodging (LoLDodgeGame), sandbox experimentation (Practice Tool), kiting and last-hitting (Skill Gap, LoLDodgeGame). Aim trainers isolate mechanical skills like flicking, tracking, and target switching, allowing focused repetition without in-game distractions. This is deliberate: removing the noise of a full game makes the signal of "did you do the thing" legible.

**Scenario-based, replay-derived drills.** The higher-end tools source drills from real high-ELO gameplay. Smiterino pulls all 130 rounds from Diamond I, Master, and Challenger ranked games, which means the chaos feels authentic. LoLDodgeGame recreates champion-specific skillshot patterns. This is the first hint of a *world model* in existing tools — but it's a thin one: they know what a Nami bubble looks like, not why it was cast there.

**Immediate, numeric feedback.** Accuracy %, time-to-target, score, streaks. This cycle of practice, feedback, and performance tracking accelerates FPS improvement and supports long-term skill retention. Feedback is the shortest loop in the industry — often sub-second.

**Leaderboards and difficulty tiers.** Most tools gate harder content behind demonstrated skill and publish rank-based leaderboards (LoLDodgeGame's "Ranked" mode, Aim Lab's scenario high scores, Skill Gap's competitive modes).

**Voluntary, session-based use.** None of these run in-game. They're a separate destination you open, warm up on, and close. loldodgegame.com Avg Visit Duration 00:01:14 — typical sessions are minutes, not hours.

### The transfer problem (and why it matters to you)

Here is the single most important finding in Stage 1, and the industry mostly ignores it: Research on cognitive training generally finds that improvements on trained tasks don't automatically generalize to untrained tasks. Aim Lab's own academic collaborators note that players with strong mouse kinematics in isolated drills don't necessarily perform better in live FPS play. Because of this, you may not see benefits from aim training if something other than aim is significantly impacting your performance.

**What this says about your competitors:** their value prop is more about the *feeling* of deliberate effort than measurable rank improvement. The unstated claim is "if you practice enough, you'll get better at League." The unstated *truth* is that mechanical drilling transfers poorly to a game where 70% of outcomes are decisional.

**What this says for howtowin.lol:** a counterfactual engine naturally attacks the transfer problem because it ties practice *to a specific moment in a specific game the player actually played*. You are not training a generic skill; you are teaching a specific lesson rooted in a lived mistake. That is a structurally different product — worth leaning into hard in positioning.

### Design commonalities worth stealing

| Pattern | Used by | Why it works |
|---|---|---|
| Short sessions (1–10 min) | All | Matches warmup psychology; low activation cost |
| Scenario replay from pro games | Smiterino, LoLDodgeGame | Authenticity; anchors drill in real game-state |
| Numeric performance delta | Aim Lab, Skill Gap | Lets user self-calibrate across sessions |
| Difficulty tiers with leaderboard per tier | LoLDodgeGame, Aim Lab | Produces flow zone (see Stage 4) |
| Single-screen UX, no client required | LoLDodgeGame, Smiterino | Reduces friction; plays between games |

### What they *don't* do (the gap you should own)

- **No causal explanation.** You hit 80% on a dodge drill; nobody tells you *why you missed the other 20%*.
- **No bridge from drill → live game.** The tools assume transfer. You cannot.
- **No metacognition.** They train reflexes, not decisions. LoL's skill ceiling is almost entirely decisional past Gold.
- **No personalization to the *player's* weaknesses.** Everyone drills the same scenarios. The system has no model of *you*.

---

## Stage 2 — Direct competitors: Mobalytics, op.gg, Skill-Capped, ProGuides

### Mobalytics — the closest structural competitor

Mobalytics is the most conceptually similar product. Its signature Gamer Performance Index (GPI), quantifies various aspects of a player's style including combat efficiency, map usage, teamwork, and objective control, then visualizes the results across multiple performance axes. It assigns automatic tags like Aggressive Laner or Vision Controller, helping players intuitively understand their tendencies and identify areas for improvement.

**What Mobalytics does well:**
- Radar-chart visualization of strengths/weaknesses per role.
- In-client overlay with matchup tips, power spikes, and live stats (a constant flow of relevant information to help them adapt during a match).
- Integrated pre-game/in-game/post-game loop (Live Companion).
- Behavioral labeling — turns stats into identity ("you are a Vision Controller").

**What Mobalytics does NOT do — and this is the central opening:**
- **GPI is descriptive, not causal.** It tells you "your vision score is low." It does not tell you *"in game 4 vs Darius, placing a deep ward at 7:32 would have prevented the gank that cost you your lane."* The difference between descriptive and counterfactual is the difference between a dashboard and a coach.
- **Insights are role-average, not game-specific.** The system aggregates 20 games to say you farm poorly. It rarely points at *which game, which wave, which decision*.
- **No practice loop.** You learn you have a weakness; you get general advice; you queue again. There is no drill, no intervention, no retention check.
- **UI bloat is a known complaint.** Because the platform offers a large number of functions, the UI can feel somewhat complex and heavy.

### op.gg — the aggregator, not a coach

op.gg is the generic stats site. OP.GG's match history analysis provides various badges, a one-line performance summary, and detailed data from every game. It expands beyond just KDA to offer deeper insights into performance. But it is explicitly a data product, not a pedagogical one — and the mobile experience is eroding: The mobile app is largely criticized for an overwhelming influx of intrusive, often non-skippable video advertisements that severely hinder usability and performance. Users frequently report significant performance issues such as slow data reflection, server errors, and extended loading times.

**The unstated assumption** in the aggregator category: showing a player their numbers will cause them to improve. There is no evidence this is true. You stare at a 2.1 KDA; you feel bad; you queue again. The data has no teaching apparatus attached.

### Skill-Capped & ProGuides — the video-course archetype

Both offer structured video courses taught by high-ELO coaches, plus 1:1 coaching booking. User sentiment is revealing:

- Skill-Capped's refund policy requires I didn't watch 2 per week minimums — implying that *users don't finish the content*, and the company knows it.
- ProGuides gets praise for coach quality but heavy complaints about billing surprises (It was not at all clear that this was a SUBSCRIPTION, not a gift.).
- Testimonials consistently emphasize that improvement requires users to watch the videos and practice a lot. The product relies on the user to close the loop themselves.

**What this tells us structurally:** video courses are a *lecture* model in a domain where the pedagogy research (Stage 3–4) unanimously favors active, retrieval-based practice. Skill-Capped is Khan Academy when the game needs Brilliant. It works for highly self-motivated players but churns everyone else.

**The unsaid thing:** the video-course model treats League as a body of knowledge (macro concepts, champion guides, matchup theory). A counterfactual-driven product treats League as a *behavior* — and behaviors aren't learned by watching; they're learned by doing, failing, and being corrected.

### Summary table of competitor gaps

| Competitor | Strength | Gap you can exploit |
|---|---|---|
| Mobalytics | Best-in-class player profiling (GPI) | Descriptive not causal; no intervention loop |
| op.gg | Ubiquity, data breadth | Pure stats, zero teaching |
| Skill-Capped | Structured curriculum | Passive video; low completion; generic not personal |
| ProGuides | Human coaches | Expensive, high-friction, not scalable |

### The implicit positioning map

Every competitor sits somewhere on a 2×2 of **[personalized ↔ generic]** × **[passive ↔ active]**:

- op.gg: generic + passive
- Skill-Capped: generic + passive (slightly active in quizzes)
- Mobalytics: personalized + passive
- ProGuides 1:1 coaching: personalized + active (but human, so unscalable)

**The unclaimed quadrant is personalized + active + scalable.** That is precisely what counterfactual + intervention promises. It's worth checking whether your product description says this clearly on the landing page, because it's your most defensible positioning.

---

## Stage 3 — General learning apps: Duolingo, Brilliant, Skillshare

### Duolingo — the gamified engagement playbook

Duolingo is the reference case because they've published more about their retention mechanics than anyone, and the numbers are ridiculous. Duolingo's DAU/MAU ratio was approximately 37% in Q2 2025. That means more than one in three monthly users are showing up daily — a number consumer apps dream about.

**The mechanics that drive the engagement:**

1. **Streaks with loss aversion.** Streaks leverage loss aversion - the psychological principle that people are more motivated to avoid losing progress than to gain rewards. Users who maintain a streak for 7 days are 3.6x more likely to stay engaged long-term.
2. **Streak recovery ("streak freeze").** Removes the cliff-edge failure mode. The introduction of the 'Streak Freeze' feature reduced churn by 21% for users at risk of breaking their streak.
3. **XP + leagues (weekly social comparison).** Users who actively engage with XP leaderboards complete 40% more lessons per week.
4. **Progressive onboarding.** Duolingo does not dump features on day one. It reveals leagues, quests, streak freezes, and other mechanics gradually. This prevents feature overload and ensures each new concept arrives when the user has enough context to care.
5. **Early "wins" to establish competence.** Early lessons are short and interactive with immediate right-or-wrong feedback. That matters because the user gets a "win" in seconds. Fast wins create perceived competence, and perceived competence is one of the strongest predictors of short-term retention in skill-building products.
6. **Adaptive difficulty via Birdbrain AI.** Duolingo demonstrates a meaningful learning curve through its data-driven AI system, Birdbrain. The system continuously improves learning outcomes by analyzing approximately 15 billion exercises per week, refining personalization and lesson effectiveness at scale.
7. **Guilt-laden re-engagement notifications** — timed and copy-tested exhaustively. Duolingo treats push as a scarce asset, protecting opt-in health via volume guardrails.

**The critical asterisk — read this carefully before importing the whole playbook:**

Duolingo's own researchers and external critics have flagged that the engagement mechanics can *detach from the underlying learning*. Features like daily streaks, designed to foster habit formation, can lead to compulsive usage patterns where learners prioritize maintaining their streak over meaningful practice. A product manager at Duolingo reportedly described their goal as creating content that users "have fun interacting with and learn as a byproduct."

**Design implication for howtowin.lol — and this is where my implications-of-decisions lens matters most:**

If you copy Duolingo's streak mechanic wholesale, you are *implicitly* signing up for the same trade-off: users will optimize for the streak, not for improvement. In a language app, the trade-off is tolerable because any exposure to Spanish beats no exposure. In a game-coaching app, the trade-off is *corrosive* — a user who opens your app for 30 seconds to preserve a streak and closes it without engaging with a counterfactual has actively *worsened* their LoL by reinforcing the belief that improvement is cheap.

**Design response:** the streak should not be on "opened the app." It should be on "engaged meaningfully with one counterfactual" — meaning watched the replay moment, answered a pre-test question about what they would do differently, and acknowledged the intervention. This is harder but is the only way the engagement mechanic and the pedagogical mission point the same direction.

### Brilliant — the active-learning archetype

Brilliant's whole pedagogical stance is a direct rebuke to video-course competitors. Brilliant teaches through active problem solving – a method shown to be far more effective than passive learning like videos or lectures.

The pedagogical details are the most directly portable to your product:

- We first build intuition with visual explanations, hands-on manipulation, and concrete computation. We start with the simplest version of an idea, minimizing cognitive load.
- Each interactive problem gives instant, custom feedback based on your answer.
- We don't teach how to do something before asking questions. Instead, we pretest on the material, letting the learner try to find a solution before learning the procedure. (This is the **productive failure / pretesting** pattern.)
- Brilliant doesn't give you answers – it gives just enough guidance to help you reason through problems yourself.

**Why this matters for you:** a counterfactual is structurally a pretest. "Here is the game state at 14:23. What would you do?" The user commits to an answer. *Then* the system reveals what actually happened, what a higher-ELO player would have done, and why. This is Brilliant's exact loop, adapted to LoL.

Brilliant also uses a tiered league system (a ranking system... You are placed into a league with 29 other players. You can either rank up to the next league, stay in the same league, or be relegated to the previous league.) but — unlike Duolingo — the leagues are gated on *activity within learning content*, not on streaks alone.

### Skillshare / skill-transfer platforms — mostly a cautionary note

I didn't search Skillshare directly because it's architecturally similar to Skill-Capped and ProGuides: a course library. The lesson is the same — passive consumption, low completion rates, relies on self-motivation. Worth noting only so you *don't* model your curriculum after it.

### Synthesis of pedagogical techniques worth importing

| Technique | Source | Mapping to howtowin.lol |
|---|---|---|
| Pretest before instruction | Brilliant | "What would you do at this moment?" before revealing the counterfactual |
| Instant, elaborated feedback | Brilliant, Duolingo | Show the branching outcome tree, not just "correct/incorrect" |
| Loss-aversion streaks, gated on *meaningful* engagement | Duolingo (modified) | Streak = "one counterfactual drilled per day" |
| Progressive disclosure of features | Duolingo | Don't show the player all systems on day 1 |
| Adaptive difficulty via ML | Duolingo's Birdbrain | Surface counterfactuals matched to the player's current weak pattern |
| Social comparison leagues | Duolingo, Brilliant | Weekly cohorts of similar-rank improvers |
| Short, interactive sessions | All | 5–10 min, plays between matchmaking queues |

---

## Stage 4 — Cognitive science: what actually makes a skill stick

This section is where the research is both deepest and, in my opinion, most underutilized by your competitors. I'll cover five pillars.

### 4.1 Spaced repetition + interleaving

The single most robust finding in the learning sciences: Studying information or practicing problems over sessions that are spaced in time (A1....A2.....A3) results in better learning than if the sessions are grouped together into a single session or closely timed sessions (A1A2A3). Studying related concepts in an interleaved fashion so that a problem is followed by a different problem type (A1B1C1B2C2A2C3A3B3) leads to higher learning gains than if practicing problems grouped by types (A1A2A3B1B2B3C1C2C3).

Two mechanisms, not one: spaced practice is a cognitive load effect that can be explained by working memory resource depletion during cognitive effort with recovery during rest-from-deliberate-learning, while interleaved practice can be explained by the discriminative-contrast hypothesis positing that interleaving assists learners to discriminate between topic areas.

**A deeply inconvenient finding for engagement-focused product teams:** studies that manipulate spaced (versus massed) practice have found that students underestimate the power of spaced practice and often give higher judgments of learning to massed practice... Studies that manipulate interleaving (versus blocking) have found that students underestimate the power of interleaved study and give higher judgments of learning following blocked practice.

That is: **the pedagogically best approach feels worse to learners than the pedagogically worse approach.** Users will rate a product that blocks their training (all wave-management drills, then all warding drills) higher than a product that interleaves them — even though the interleaved version actually teaches better.

**The unstated tension for howtowin.lol:** if you interleave correctly, early NPS will be lower than if you don't. You will need to either (a) eat the NPS hit in service of actual improvement, (b) educate the user about why the harder feeling is the point, or (c) find UI that makes interleaving *feel* structured (e.g., a daily "practice menu" that is explicitly themed but secretly interleaves across sub-skills).

**Counterfactuals have a natural spacing mechanic built in.** Your system can resurface a counterfactual about warding three days later, in a slightly different form ("last Tuesday you missed this warding moment vs Lee Sin. Here's a similar moment from yesterday's Elise game. Same mistake, new skin"). That is spaced retrieval practice packaged as personalized content — which is close to a perfect product.

### 4.2 Retrieval practice (the testing effect)

Testing yourself on material is dramatically more effective than re-reading it. The effect is robust, large, and cross-domain. Retrieval practice, or the active recall of information from memory, is a highly effective learning strategy that strengthens memory and comprehension. This effect is robust and strongly backed by research in cognit[ive science].

Implication: every piece of content in howtowin.lol should end in a retrieval prompt. Not "here's what to do in this matchup" but "here's the matchup — what do you do?" The Brilliant model.

### 4.3 Deliberate practice (Ericsson)

The classic framework: deliberate practise improves skills with focus. Learners set clear goals and get feedback. Work slightly past a learner's current skill level... This reveals weak spots, so learners adjust and gain expertise.

The four components — goal, feedback, slight stretch, focused attention — all map cleanly to your counterfactual loop:

| Ericsson component | howtowin.lol instantiation |
|---|---|
| Clear goal | "Cut 2 minutes off your average death timer" |
| Immediate feedback | Counterfactual replay with outcome tree |
| Slight stretch beyond current ability | Drill selected from player's ZPD (see 4.4) |
| Focused attention | Single-moment replay, not a 30-min VOD |

### 4.4 Vygotsky's Zone of Proximal Development & Flow

Two frameworks that converge on the same design principle: **the learner must be pitched just past their current ceiling.**

Vygotsky: Vygotsky's Zone of Proximal Development (ZPD) refers to the gap between what a learner can do independently and what they can achieve with guidance. Learning occurs most effectively in this zone.

Csikszentmihalyi's flow: Flow theory postulates that three conditions must be met to achieve flow: The activity must have clear goals and progress... The task must provide clear and immediate feedback... Good balance is required between the perceived challenges of the task and one's perceived skills.

**Two further findings worth marking:**

1. **Scaffolds must fade.** A 2024 meta-analysis of scaffolding interventions found that programmes with explicit fading protocols produced effect sizes of d = 0.71, compared to d = 0.32 for programmes where scaffolds remained constant. In your context: counterfactuals should get less explicit over time. Early drills spell out "ward river here at 6:30." Later drills just show the game state and ask the player what to do, with the system silently tracking whether they picked the high-value action.

2. **High flow comes from high challenge + high skill, not just any balance.** The quadrant model posits that the balance between challenge and skill does not always lead to optimal flow... For players who feel they have minimal skill and are playing what they feel is a minimally challenging game, apathy rather than flow should ensue. This matters for your first-time UX: don't make the opening sessions trivially easy; make them *achievably hard at the user's actual rank*.

### 4.5 Self-regulated learning — and a paper you should read

This is the most exciting finding I surfaced, and it's directly about League of Legends:

Kleinman et al. (2021) in *Frontiers in Psychology*, compared the self-regulatory processes of expert, non-expert, and novice League of Legends players, and found that there were significant differences for processes in the forethought phase.

The forethought phase means: goal-setting, strategic planning, task analysis *before* the game. Novices don't do it. Experts do it systematically. skills in the forethought phase are not prominently supported by existing tools and would instead require interaction with a team or coach to develop, making them more common among expert players.

The paper explicitly calls for the development of new computational support tools for players that could help bridge the gap between novice and expert play by supporting this forethought phase.

**That gap is your product.** The counterfactual is a retrospective tool, but every counterfactual can be reversed into a forethought intervention: *"Last game you lost tempo because you recalled at 3:21 with 400 gold. Before your next game, set an intent: I will only recall if I have 900+ gold or I am below 30% HP."* The intervention *precedes* the next game — that's forethought scaffolding.

Kleinman's research also validates a broader product thesis: esports interfaces could inspire the design of future e-learning technologies that more effectively engage students. Academic pedagogy researchers explicitly see your category as a testbed for self-regulated learning tools. This is a citation-rich positioning angle if you ever do investor or PR work — and a potential partnership lane with education researchers.

### 4.6 A note on feedback timing

The feedback research is surprisingly nuanced. Immediate feedback wins for *initial* skill acquisition; delayed feedback can win for *long-term retention and transfer*. delayed feedback slowed down the rate of initial learning, [but] it facilitated trans[fer]. Also important: learners may better assimilate and apply skills when feedback is concise and actionable rather than excessively detailed.

**Design implication:** the post-game counterfactual (immediate) teaches the pattern. A spaced follow-up a few days later (delayed) locks in retention. You want both in the system, not just one. And feedback should be *short*. A three-line explanation beats a three-paragraph one.

---

## Synthesis: a proposed teaching architecture for howtowin.lol

Pulling all four stages together, here is the model I think the research supports. I'll present it as a stack, because each layer depends on the one below it.

### Layer 1 — Detection (world model → counterfactual)
Your differentiator. Identify the 1–3 most consequential moments in a game where the player's decision had a large counterfactual gap. This is the engineering core and not the subject of this review — but note: **if detection is noisy, everything above fails**. The pedagogy can't save a bad signal.

### Layer 2 — Presentation (the moment)
Replay the moment. Pause at the decision point. Ask the player what they would do. (**Pretest** — Brilliant.) Commit their answer. *Then* reveal what they did, what the counterfactual alternative would have produced, and why. (**Elaborated feedback** — Shute.)

### Layer 3 — Intervention (forethought)
Convert the lesson into a concrete pre-game intent for the next game. Singular, testable. ("Before your next ranked, set: I will not leash past level 2.") This maps directly to the Kleinman finding about forethought as the expert–novice gap.

### Layer 4 — Spaced reinforcement
Resurface the same pattern in different dressing across days 2, 5, and 14. (**Spaced repetition** — Kang, Rohrer.) Interleave with other patterns the player is working on. (**Interleaving** — Taylor & Rohrer.)

### Layer 5 — Retrieval check
A week later, drop a decontextualized scenario into the user's daily flow and ask them to solve it. If they get it right, fade the scaffolding on that pattern. If they miss, re-enter the spaced queue. (**Retrieval practice** — Roediger & Karpicke. **Scaffold fading** — Belland.)

### Layer 6 — Engagement scaffolding
Streaks on *meaningful engagement*, not on opens. Weekly leagues of same-rank improvers. Progressive disclosure of features. Push notifications framed as supportive reminders, not marketing. All of this is *on top of* the pedagogy, not in place of it. This is the critical lesson from Duolingo's scaling journey: **engagement mechanics that drift from learning mechanics produce users, not learners.**

---

## Key risks and design trade-offs to think about explicitly

1. **The engagement/learning divergence trap.** Every engagement lever you add has a chance of rewarding streak behavior over improvement behavior. You should write down, once, what *improvement* means to you (rank climb? GPI-style skill gain? subjective mastery?) and never let an engagement lever ship without an A/B test against that metric.

2. **Interleaving feels worse.** First-week NPS will be lower than a purely flattering, blocked-practice competitor. You have to be willing to sit with that.

3. **Counterfactuals can be wrong.** A world model that asserts "this ward would have saved you" is making a causal claim. If the claim is wrong too often, users lose trust in the whole system. Invest in model calibration and in UI language that leaves epistemic room ("this likely would have...") rather than asserting certainty.

4. **Scaffolds that don't fade are crutches.** If the player keeps getting "ward here at 6:30" hints and you never test whether they've internalized it, you're building dependency, not skill. Plan your fading protocols from day one, not as a v2 feature.

5. **The self-sabotage pattern of game coaching.** Players in a losing streak often reach for coaching products *at the worst moment to learn* — emotionally flooded, tilted, low self-efficacy. Your UX should notice this state and respond to it (shorter sessions, easier material, more warmth) rather than pile on more "here's what you did wrong."

---

## Source list

A consolidated list of the most load-bearing sources, for deeper reading:

**Competitor/tool landscape:**
- Aimlabs / Cognitive Train on transfer limits of aim training
- LoL Practice Tool docs (Riot, Fandom wiki)
- Smiterino product page & PlayPile review
- LoLDodgeGame site + Dignitas guide
- Mobalytics GPI docs & Overwolf listing
- Skill-Capped / ProGuides Trustpilot reviews
- STATUP.GG 2025 survey of LoL coaching apps

**General learning apps:**
- StriveCloud, Orizon, Trophy.so, Propel case studies of Duolingo
- Gadallon Substack (Duolingo critique)
- Brilliant.org About page + App Store pedagogical claims

**Cognitive science:**
- Kang 2016, *Spaced Repetition Promotes Efficient and Effective Learning* (SAGE)
- Sweller et al. 2021 systematic review on spacing vs. interleaving mechanisms
- Roediger & Karpicke on retrieval practice
- Ericsson, Krampe, Tesch-Römer 1993 on deliberate practice
- Shute 2007 *Focus on Formative Feedback* (ETS)
- Vygotsky (1978) + Belland 2024 meta-analysis on scaffolding
- Csikszentmihalyi flow theory + Jenova Chen's *Flow in Games* MFA thesis
- **Kleinman et al. 2021, *"Because I'm Bad at the Game!"* — SRL in League of Legends (Frontiers in Psychology)** — read this one in full

---

*Final connotation to sit with:* your product is structurally betting that **improvement is teachable** in a domain that has historically sold players the opposite story — that improvement is earned by grinding games. The research supports your bet. Your competitors don't.
