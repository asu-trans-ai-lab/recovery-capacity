# Recovery Capacity / LEA Model

This repository accompanies the accepted manuscript:

**From Traffic-Flow Capacity to Recovery Capacity: An Episode-Based Framework for Multi-Sector Urban Resilience**

*Transportation Research Part C* (TRC-25-03191R1; VSI: In memory of H. Mahmassani)

Authors: Fang Tang, Xuesong Zhou, Edward Zhu, Irfan Batur, Yudai Honma, Ram Pendyala, Cynthia Chen


## Overview

Urban systems experience disruptions that differ in magnitude, duration, and final recovery outcome. Existing resilience measures often emphasize performance loss, recovery duration, or trajectory area, but they do not directly represent the peak rate at which a disrupted system recovers.

Inspired by the traffic-flow view of capacity as a dynamic, state-dependent peak rate, this paper develops an **episode-based recovery-capacity framework**. Key components include:

- **Double Quadratic Queue (DQQ)**: an episode filter that identifies critical transition points in disruption–recovery trajectories
- **Linear-Exponential Adaptation (LEA)**: represents the recovery trajectory through the instantaneous healing rate \(h(t)\), cumulative healing capacity \(H(t)\), and recovery fraction \(F(t)\)
- **Single-period recovery capacity**: defined as the maximum healing rate, \(R_{cap} = ab\,e^{-c}\)
- **Multi-episode stress–strain relationships**: complementary descriptive measures of disruption sensitivity and transition loss or gain

Using Google COVID-19 Community Mobility Reports, LEA generally achieves lower fitting and prediction errors than benchmark recovery models. Extreme-heat episodes in Phoenix and post-earthquake infrastructure restoration in Japan are included as transfer illustrations with case-specific variables and interpretations. The framework is descriptive and diagnostic rather than causal.

## Repository Contents

**All LEA model data, analysis code, and related outputs from this paper are included in this GitHub repository.**

| Directory | Description |
|-----------|-------------|
| [`data/`](./data/) | Input data for the LEA model (U.S. mobility, Japan water / transportation, etc.) |
| [`codes/`](./codes/) | All code for LEA fitting, prediction, regression, and visualization |
| [`outputs/`](./outputs/) | Fitting / prediction results and statistical outputs |
| [`fig/`](./fig/) | Figure files |

Repository URL: https://github.com/asu-trans-ai-lab/recovery-capacity
