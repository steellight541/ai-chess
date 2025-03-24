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

### Vergelijkbare Projecten  

| Project       | Sterkten                  | Zwakten                          |  
|---------------|---------------------------|----------------------------------|  
| **Stockfish** | Zeer sterke engine        | Geen GUI, complexe installatie   |  
| **Lichess**   | Online AI, uitgebreide DB | Internetafhankelijk, overweldigend voor beginners |  

### Dataset(s)  

- **openings.json**:  

  - Bevat 500+ openingszetten (UCI-formaat).  
  - Bron: `NielsBotData/.Dataset/openings.json`.  
  - Verwerking: Dubbele zetten gefilterd via `Counter` uit Python-collections.  

### AI-Algoritmen  

1. **Alpha-Beta Pruning**  

   - **Reden**: Efficiëntie bij beperkte rekenkracht.  
   - **Diepte**: Maximaal 4 lagen (trade-off snelheid vs. nauwkeurigheid).  
   - **Workflow**:

     ```mermaid  
     graph TD  
         A[Start] --> B{Openingsmatch?}  
         B -->|Ja| C[Kies willekeurige opening]  
         B -->|Nee| D[Alpha-Beta Zoeken diepte 4]  
         D --> E[Evalueer zetten met PST]  
         E --> F[Selecteer beste/zufällige optimale zet]  
         F --> G[Update bord]  
     ```  

2. **Piece-Square Tables (PST)**  

   - Voorbeeld Waarden:  

     | Stuk   | Centrum (Bonus) | Rand (Malus) |  
     |--------|-----------------|--------------|  
     | Paard  | +30            | -20          |  
     | Loper  | +25            | -15          | 

   - Code:  

     ```python  
     PAWN_TABLE = [0, 0, 0, ..., 50, 50, 50]  # Positie-afhankelijke scores  
     ```  

3. **Openingsboek**:  
   - Random selectie uit JSON-dataset om variatie te garanderen.  

### Tools & Libraries  

- **PyQt5**: Voor GUI-elementen (bijv. `QSvgWidget`).  
- **python-chess**: Schaaklogica en move validation.  
- **QThread**: Asynchrone AI-berekeningen.  

### Hardware/Software  

| Component     | Specificatie               |  
|---------------|----------------------------|  
| **OS**        | Windows/Linux/macOS        |  
| **CPU**       | Intel i5 of equivalent     |  
| **RAM**       | 4GB+                       |  
| **Libraries** | PyQt5, python-chess, json  |  

---

## **Resultaat**  

### Overzicht  

![GUI Scherm](chess_gui_example.png)  
*Figuur 1: GUI met gemarkeerde zetten en AI-AI-modus.*  

### Technische Uitwerking  

1. **GUI Workflow**  

   ```python  
   def handle_square_click(self, square):  
       if self.selected_square is None:  
           self.selected_square = square  # Highlight geselecteerd stuk  
           self.update_board()  

2. **AI Prestaties**

| Meting     | waarde               |  
|------------|----------------------|  
| gem. responstijd       | 1.8sec per zet       |

## Uitbreiding

- Zelf een betere dataset maken.
- Meer data geven zodat AI meer leerd
- Algoritme verbeteren

## Conclusie

Het project is geslaagd. Er is een gebruiksvriendelijke GUI en de AI tegenstander maakt logische en goede zetten.

De subdoelstellingen zijn ook voltooid.
