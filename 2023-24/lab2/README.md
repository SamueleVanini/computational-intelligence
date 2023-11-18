Samuele Vanini 318684 Lab 2 - Nim Game
===

To test the code first install and run the poetry environment (be sure to be in the folder 2023-24)

```
poetry install
poetry shell
```

then run the main (evolutions can take a while, for a faster run modify LAMBDA and N_GEN at the end of the file)

```
python main.py
```

Information displayed are related to:
- 1 vs 1 matchs between different type of player;
- ES agent based on probability (different strategy are used and each of them has a different probability to be taken each turn of a match);
- ES agent based on adaptive rules (rule: priority on taking a certain amount of object from a heap)