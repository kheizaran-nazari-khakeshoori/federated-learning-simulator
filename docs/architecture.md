# Architecture

UI (left/right) -> Engine -> Algorithms (FedAvg/Prox/Adam) -> Model (SimpleCNN) -> Data (MNIST/Synthetic)

## Flow

```
LeftPanel (controls) --vars--> Engine.start() --weights--> Global Model --chart--> RightPanel
```

## Initial Diagram

Docs cleaned before release tag

LeftPanel -> Engine -> FedAvg -> CNN -> Data
