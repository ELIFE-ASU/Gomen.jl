# Gomen API

```@meta
CurrentModule = Gomen
```

## Games
```@docs
Game
play
```
## Rules
```@docs
AbstractRule
apply
Sigmoid
Heaviside
```

## Schemes
```@docs
AbstractScheme
decide
CounterFactual
```

## Arena
```@docs
AbstractArena
game
graph
scheme
Base.length
LightGraphs.edges
LightGraphs.neighbors
Arena
payoffs
```
