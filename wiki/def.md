An open game of type

$$\mathcal{G} : (X, S) \rightarrow (Y, R)$$

is, by definition, a 4-tuple

$$\mathcal{G} = (
  \Sigma_\mathcal{G},
  \mathbf{P}_\mathcal{G},
  \mathbf{C}_\mathcal{G},
  \mathbf{B}_\mathcal{G}
)$$

where

- $\Sigma_\mathcal{G}$ is a set, called the **strategy profiles** of $\mathcal{G}$
- $\mathbf{P}_\mathcal{G} : \Sigma_\mathcal{G} \rightarrow \mathrm{hom}_\mathcal{C}(X, Y)$ is called the **play function** of $\Sigma_\mathcal{G}$
- $\mathbf{C}_\mathcal{G} : \Sigma_\mathcal{G} \rightarrow \mathrm{hom}_\mathcal{C}(X \otimes R, S)$ is called the **coplay function** of $\Sigma_\mathcal{G}$
- $\mathbf{B}_\mathcal{G}: \mathrm{hom}_\mathcal{C}(I, X) \times \mathrm{hom}_\mathcal{C}(Y, R) \rightarrow \Sigma_\mathcal{G} \rightarrow \mathcal{P} \Sigma_\mathcal{G}$ is the **best response function** of $\Sigma_\mathcal{G}$

```haskell
instance Category (->) where
  id = Prelude.id
  (.) = (Prelude..)

instance MonoidalCategory (->) where
  unit = id
  (>*<) f g (a, b) = (f a, g b)

instance SymmetricMonoidalCategory (->) where
  swap (a, b) = (b, a)

instance OpenGameCategory (->) where
  openGame (play, coplay) = OpenGame
    { strategyProfiles = [play]
    , playFunction = play
    , coplayFunction = coplay
    , bestResponseFunction = \_ _ strategies -> strategies
    }
```

```haskell
play :: Int -> Int
play x = x + 1

coplay :: (Int, Int) -> Int
coplay (x, r) = x * r

game :: OpenGame (->) Int Int Int Int
game = openGame (play, coplay)

main :: IO ()
main = do
  let strategies = strategyProfiles game
  print strategies  -- Output: [play]

  let playResult = playFunction game 1
  print playResult  -- Output: 2

  let coplayResult = coplayFunction game (2, 3)
  print coplayResult  -- Output: 6

  let bestResponses = bestResponseFunction game id id strategies
  print bestResponses  -- Output: [play]
  ```