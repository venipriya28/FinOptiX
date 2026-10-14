# Azure Advisor: resize and shutdown recommendations

## Source
Azure Advisor cost recommendations: https://learn.microsoft.com/en-us/azure/advisor/advisor-cost-recommendations

## Retrieved guidance
Azure Advisor can recommend resizing or shutting down underutilized virtual machines. Its VM recommendations consider CPU, memory, and outbound network utilization over a configured lookback period. A resize can require a restart, and savings may differ with reservations or other purchasing commitments.

## FinOptiX interpretation
FinOptiX exposes CPU and memory constraints and flags that its dataset does not represent purchase commitments or application latency. It does not apply Azure Advisor's internal algorithm.
