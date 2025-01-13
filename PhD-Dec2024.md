## Эволюционно стабильные равновесия в играх




*Равновесие Нэша* — это `. . .`

*Смешанная стратегия* — это `. . .`

**Эволюционно стабильная стратегия** — это смешанная стратегия, "устойчивая к вторжению" новых стратегий. Представим ситуацию: есть популяция агентов, применяющих в рамках игры некую стратегию, и к ним добавляется группа "захватчиков" с альтернативной стратегией. Оригинальная стратегия будет считаться эволюционно стабильной, если она приводит к бо́льшему выигрышу в игре против стратегий "захватчиков." [1, с. 28]

В случае симметричной игры в нормальной форме с двумя игроками $G = (\{1,2\}, A, u)$ и смешанной стратегией $S$ говорят, что $S$ является эволюционно стабильной стратегией тогда и только тогда, когда для некоторого $\epsilon > 0$ и для всех остальных стратегий $S'$ выполняется [1, с. 29]:

$$u(S, (1 - \epsilon)S + \epsilon S') > u(S', (1 - \epsilon)S + \epsilon S ')$$

Используя свойства ожидания (какие???) это условие эквивалетнтно выражению:

$$(1 - \epsilon)u(S, S) + \epsilon u(S, S') > (1 - \epsilon) u(S', S) + \epsilon u(S', S')$$

$S$ является **слабой эволюционно стабильной стратегией** тогда и только когда, когда для некоторого $\epsilon > 0$ и для любого $S'$ справедливо либо:

$$u(S, S) > u(S', S),$$

либо каждое из следующих выражений:

$$u(S, S) = u(S', S)$$
$$u(S, S') \geq u(S', S')$$

Слабое определение включает в себя стратегии, в которых "захватчики" так же успешно побеждают оригинальную популяцию, как и других захватчиков. [1, с. 29]

- [ ] Что такое A, u, \epsilon??
- [ ] Что за свойства ожидания?? Это математическое ожидание?

## Рандомизация наград

https://openreview.net/forum?id=lvRTC669EY_
Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization


- [ ] Randomization in game theory

  Related to mixed strategies!!

- [ ] Рандомизация наград в теории игр
- [ ] Рандомизация и оптимальные стратегии
- [ ] Игры с рандомизированными наградами
- [ ] Эксперименты с наградами в контексте теории игр
- [ ] В каких случаях рандомизация наград полезна? В каких вредна?

### Stochastic games!
https://en.wikipedia.org/wiki/Stochastic_game

- Single player - Markov Decision Process
- Multi-player - Stochastic game

http://www.sti.uniurb.it/events/sfm10qapl/slides/kucera-sfm-qapl2010.pdf:

- the impact of players’ choices in uncertain;
- the players’ choice can be randomized.

## Новые датасеты

- Noisy datasets for machine learning
- Benchmark datasets for optimization
- Datasets with noise for machine learning
- Datasets for optimization algorithms
- Datasets with optimization challenges

## Источники

1. K. Leyton-Brown and Y. Shoham. *Essentials of game theory: a concise, multidisciplinary introduction.* Morgan and Claypool, 2008.
2. Z. Tang, C. Yu, B. Chen, H. Xu, X. Wang, F. Fang, S. S. Du, Y. Wang, and Y. Wu. Discovering diverse multi-agent strategic behavior via reward randomization. In *International Conference on Learning Representations*, 2021.