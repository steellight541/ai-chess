# Verslag: Chess AI met PyQt5 Interface  

---

## **Doelstelling(en)**  

### Hoofddoelstelling  

Ontwikkelen van een schaakapplicatie met een AI-tegenstander, gekenmerkt door een gebruiksvriendelijke interface en een competente AI voor casual spelers.  

### Subdoelstellingen  

1. Implementeren van een visueel aantrekkelijk bord. 
2. Ontwerpen van een AI die gebruikmaakt van positionele evaluatie en openingskennis.  
3. Mogelijkheden bieden voor zowel mens-AI als AI-AI interactie.  
4. Garanderen van correcte schaakregels en move validation.  

---

## **Probleemstelling**  

Het project richt zich op het maken van een AI adhv van machine learning. Deze AI moet voldoen aan de regels van schaken en moet nuttige stappen zetten. Er moet gebruik gemaakt worden van machine learning door bijvoorbeeld openingen geven om de AI op weg te helpen in de start. 

**Doelgroep:**  

- Studenten en docenten aan VIVES
- bezoekers op opendeurdag 

---

## **Analyse**  

### Vergelijkbare Projecten:  

- **Stockfish**: Open-source engine met geavanceerde zoekalgoritmen.  
- **Leela Chess Zero**: Gebruikt neurale netwerken en reinforcement learning.  

### Dataset:  

- Een JSON-bestand (`openings.json`) met openingszetten in UCI-formaat, verkregen via historische schaakpartijen.  

### AI-Algoritmen:  

1. **Alpha-Beta Pruning**:  
   - Gekozen voor snelle zoekoperaties met een diepte van 4 lagen.  

2. **Piece-Square Tables**:  
   - Positionele evaluatie via vooraf gedefinieerde scoretabellen per stuktype (bijv. pionnen scoren hoger in het midden).  

### Tools en Libraries:  

- **Python-Chess**: Voor schaakregels en bordvisualisatie.  
- **PyQt5**: Grafische interface.  
- **QThread**: Asynchrone AI-berekeningen om de GUI responsief te houden.  

### Hardware/Software:  

- Geen GPU-vereisten door beperkte zoekdiepte.  
- Cross-platform (Windows/Linux/macOS) via Python.  
- Deployment als standalone script (geen Docker of `.exe`).  

---

## **Resultaat**  

### Overzicht:  

De applicatie combineert een interactieve GUI met een regelgebaseerde AI. Gebruikers kunnen zetten invoeren via UCI of drag-and-drop, terwijl de AI reageert met openingsdatabase-gebaseerde zetten of alpha-beta-berekeningen.  

### Technische Uitwerking:  

#### GUI:  

- `ChessBoardWidget` toont het bord via SVG en detecteert muisinteracties.  
- Ondersteuning voor promotie en zichtbare legale zetten (Figuur 1).  

```python
class ChessBoardWidget(QSvgWidget):  
    pieceMoved = pyqtSignal(int, int)  
    def mousePressEvent(self, event): ...  
```

#### AI-Kern:  

- **Openingsfase**: Random selectie uit `openings.json` om variatie te garanderen.  
- **Midgame**: Alpha-beta pruning met positionele evaluatie (Tabel 1).  

```python
PAWN_TABLE = [0, 50, 10, 5, ..., 0]  # Positionele scores voor pionnen  
```

#### Prestaties:  

- **Zoeksnelheid**: ~100 zetten/seconde bij diepte 4.  
- **Beperkingen**: Geen endgame-databases of neurale netwerken.  

---

## **Uitbreiding**  

### Niet-geïmplementeerde features:  

1. **Machine Learning**: Trainen van een neuraal netwerk voor evaluatiefuncties.  
2. **Diepere Zoeklagen**: Optimalisatie met transpositietabellen of parallelle verwerking.  
3. **Multiplayer**: Online integratie via chess.com API.  
4. **Educatieve Modus**: Uitleg van AI-beslissingen in realtime.  

---

## **Conclusie**  

Het project slaagt in zijn hoofddoelstellingen: een werkende schaakbot met GUI en basis-AI is gerealiseerd. De alpha-beta pruning en openingsdatabase zorgen voor een aanvaardbaar speelniveau, hoewel de evaluatiefunctie simplistisch blijft vergeleken met commerciële engines. De applicatie lost de probleemstelling op als educatief hulpmiddel, maar is niet geschikt voor competitief gebruik.