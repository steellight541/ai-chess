
# Machine Learning Report: Nigel Chess Bot

## Overzicht modelarchitectuur

- **Modeltype**: `RandomForestClassifier` (150 bomen, max. diepte=12)
- **Hybride aanpak**: combineert supervised learning met game tree search
- **Feature Scaling**: StandardScaler toegepast op invoerfeatures

## Feature Engineering

12 handgemaakte schaakspecifieke features die positionele dynamiek vastleggen:

| Feature                 | Beschrijving                            | Berekeningsmethode                            |
| ----------------------- | --------------------------------------- | --------------------------------------------- |
| Materiaalbalans         | Puntenverschil in stukken               | Som van stukwaarden (wit - zwart)             |
| Centrumcontrole         | Dominantie van centrale velden          | Aantal stukken op d4/d5/e4/e5                 |
| Stukmobiliteit          | Mogelijke bewegingsopties               | Aantal beschikbare legale zetten              |
| Veiligheid koning       | Aanvals-/verdedigingsbalans bij koning  | Aanvaller/verdedigerverhouding in koningszone |
| Pionstructuur           | Kwaliteit van de pionformatie           | Straffen voor dubbele/geïsoleerde pionnen    |
| Bedreigingen            | Direct aanvalspotentieel                | Aantal beschikbare controlezetten             |
| Ontwikkelde stukken     | Voortgang in stukactivering             | Aantal verplaatste niet-pionstukken           |
| Veroveringskans         | Mogelijkheid tot materiële winst       | Aantal beschikbare veroveringszetten          |
| Hangende stukken        | Onverdedigde kwetsbare stukken          | Aantal aangevallen onverdedigde stukken       |
| Herhaling van zetten    | Redundantie van spelstatus              | Positievoorkomensteller                       |
| Positiegeschiedenis     | Bewegingsdiversiteit                    | Unieke zetten in recente geschiedenis         |
| Nabijheid van de koning | Aanvallende druk op vijandelijke koning | Afstandgewogen stuknabijheid                  |

## Trainingsproces

1. **Gegevensvoorbereiding**:

- Verwerk PGN-gamebestanden in bordstatus-/zetparen
- Genereer kenmerkvectoren voor elke positie vóór de zet
- Gebruik daadwerkelijk gespeelde zetten als trainingslabels

2. **Trainingskenmerken**:

- Geen expliciete trein-/testsplitsing weergegeven in huidige implementatie
- Random forest verwerkt categorische zetvoorspellingen
- Kenmerken geschaald naar nul gemiddelde/eenheidsvariantie

## Voorspellingsstrategie

1. **Tactische controle**:

- Onmiddellijke schaakmatdetectie
- Waardegebaseerde vangstevaluatie

2. **Hybride besluitvorming**:

```python
if random() < 0.5:
use_search_result()
else:
use_model_prediction()
```

3. **Probabilistische selectie**:

* Combineer modelwaarschijnlijkheden (70%) met evaluatiescores (30%)
* Softmax-gewogen willekeurige selectie uit de top 3 kandidaten
* Temperatuur bemonstering voorkomt deterministisch spel

## Evaluatiemetrieken

* **Positie-evaluatiefunctie** combineert:
* Materiaalbalans (+/- stukwaarden)
* Kenmerkgewichten (bijv. +0,2/centrale controle-eenheid)
* Tactische overwegingen (controles, vastleggingen)
* **Zoekfunctie**:
* 2-laags minimax-zoekopdracht
* Evaluatiefunctie als schatter van bladknooppunten

## Sterke en zwakke punten

**Sterke punten**:

* Interpreteerbaar kenmerksysteem
* Hybride aanpak balanceert berekeningssnelheid/diepte
* Random forest verwerkt niet-lineaire relaties
* Positieherhalingstracking voorkomt 3-voudige trekkingen

**Zwakheden**:

* Beperkte diepte in zoekfunctie
* Geen implementatie van validatiemetrieken
* Handmatige kenmerkengineering beperkt complexiteit
* Geen expliciete detectie van trekkingen in evaluatie

## Verbeteringsmogelijkheden

1. **Modelverbeteringen**:

* Experimenteer met gradiëntversterkte bomen/neurale netwerken
* Monte Carlo-boomzoekopdracht toevoegen (MCTS) integratie
* Implementeer reinforcement learning framework

1. **Gegevensverbeteringen** :

* Integreer grootmeesterspeldatabases
* Voeg synthetische training toe via self-play
* Implementeer juiste trein-/validatie-/testsplitsingen

1. **Feature Engineering** :

* Voeg rokaderechtentracking toe
* Implementeer gepasseerde piondetectie
* Voeg tempo-/initiatiefmetrieken toe

1. **Evaluatie-upgrade** :

* Functie voor evaluatie van neuraal netwerk
* Eindspelspecifieke evaluatietabellen
* Verbeterde heuristiek voor trekdetectie

## Conclusie

De Nigel Chess Bot demonstreert effectieve integratie van traditionele schaakheuristiek met moderne machine learning-technieken. Hoewel het momenteel vertrouwt op klassieke feature engineering, biedt de architectuur duidelijke paden voor integratie met geavanceerdere neurale netwerkbenaderingen. De hybride voorspellingsstrategie biedt een evenwichtig compromis tussen computationele efficiëntie en strategische diepgang.
