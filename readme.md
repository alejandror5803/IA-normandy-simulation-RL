

idea principal:

sistema multi-agente 100% en gymnasium y usando Q-Learning simple (no DQN ni otras opciones) en el que cada agente debera de controlar 1 peloton individual (habiendo 4 en el equipo azul) y luego un agente que gestione a los 4 agentes de cada peloton (1 agente q controla 4 agentes).

la mision de estos agentes es tomar los puntos A,B y C del mapa de normandia, al mismo tiempo que luchar contra el equipo rojo, el equipo azul tiene una inferioridad de 1:3

Pelotones:
- formado por 5 tanques
- si hay 5 tanques y la vida de cada uno es 100, el peloton tiene 500HP si este baja de 100HP se asume que uno de los que lo formaban ha sido destruido y la capacidad de fuego disminuye de 5 a 4 tanques haciendo menos daño
- cada tanque hace un determinado daño, este se multiplica por el nº de tanques que forman el peloton
- se tienen 100 de gas y 15-20 de municion por tanque
- deben de evitar los obstaculos que se generan en el mapa
- deben recoger los suministros

Puntos:
- El pto B es el mas valioso
- A y C mismo valor
- en los puntos se podran recoger suministros
- tienen un limite de suministros 1000 de gas y 50 de municion

Equipo Rojo:
- de primeras estara hardcodeado, luego se intentara que fuese otro agente
- la entidad que lo represente en el grid2d debe usar el resource sherman.png

Equipo Azul:
- 4 pelotones manejados cada uno por un agente singular (4 agentes)
- la entidad que lo represente en el grid2d debe usar el resource tiger.png

Mapa:
- habran 3 puntos de interes (A,B,C)
- se generaran obstaculos de manera aleatoria y procedural en el mapa
- el grid 2D 25x25 celdas usando pygame y el resource mapa.png como fondo

Acciones Pelotones:
   0 = MOVE_NORTH        (mover hacia el norte)
   1 = MOVE_SOUTH        (mover hacia el sur)
   2 = MOVE_EAST         (mover hacia el este)
   3 = MOVE_WEST         (mover hacia el oeste)
   4 = ATTACK_NEAREST    (atacar al enemigo más cercano en rango)
   5 = TAKE_COVER        (buscar celda de cobertura adyacente)
   6 = RESUPPLY          (solicitar suministros si está en zona base)
   7 = HOLD_POSITION     (mantener posición y esperar)

Tipos de estructura y obstaculos del mapa:
Tipo Celda	Valor Cobertura	Penalización Mov.	Descripción
OPEN	0.0	0	Terreno abierto — sin cobertura
BUSH	0.3	1	Arbustos — cobertura ligera
FOREST	0.6	2	Bosque — cobertura moderada
RUBBLE	0.5	1	Escombros — cobertura buena
WALL	0.9	3	Muro — cobertura alta
WATER	0.0	99	Agua — infranqueable




Posible implementacion de Perbatin (Si sobra tiempo) [nivel de locura de las tropas]



CONTENIDO DE LA PAGINA DEL CANVA:
La practica se basara en la simulación de la batalla de Normandía mediante RL, el equipo azul (los alemanes) comenzaran siendo el equipo que aprenda mediante RL.

 

El agente del equipo azul, debe gestionar las cadenas de suministro para reabastecer a los tanques, gestionar los pelotones de tanques, dividir la fuerza inicial (20 tanques) en pelotones, tomar los puntos A,B,C del mapa, siendo B el mas valioso debido a su posición estratégica. Debe de poder defender a sus tropas/pelotones, atacando a las fuerzas enemigas y escondiéndose detrás de coberturas apropiadas.

 

Una vez que el agente del equipo azul aprenda y funcione bien, la idea es trasladar esto mismo al equipo rojo, para simular la batalla de caen mediante 2 agentes que controlan a un equipo cada uno usando RL.

 

El equipo azul (Alemania) tiene una desventaja de que parte con una inferioridad de 3:1 según datos históricos -- (por cada tanque alemán hay 3 tanques aliados)

Los tanques Alemanes deben de tener mas blindaje/vida que los aliados (datos históricos)

 

Dado que son muchas decisiones las que debe de tomar, la estructura agentica sera la siguiente:

 

Peloton individual:

Agente de ataque
Agente de captura
Agente de suministros
Agente de cobertura/defensa
Agente Comandante 
Estrategia:

Agente de dominacion
Agente de status (como esta cada peloton)
Agente Principal (field marshal) (Toma la decision final basada en lo comunicado por cada agente)
 