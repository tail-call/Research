--- MAGICAL HASKELL chapter II
{-# OPTIONS_GHC -Wno-overlapping-patterns #-}

data MyMaybe a = MyJust a | MyNothing

-- typeFunction Maybe (a : Type) = Just (x : a) + Nothing ()

-- divide :: Double -> Double -> Maybe Double
-- divide x 0 = Nothing
-- divide x y = Just (x / y)

divide :: (Eq a, Fractional a) => a -> a -> Maybe a
divide x y = case y of
    0 -> Nothing
    _ -> Just (x/y)


square :: Num a => a -> a
square r = r * r

divideSquare x y = maybe 0 square (divide x y)

divideSquare x y = let r = divide x y in
    case r of Nothing -> Nothing
              Just z -> Just (square z)


data Person where
    PersonConstructor :: String -> Int -> Person

x :: Person
x = PersonConstructor "Man" 1


-- ghci> (*2) <$> (Just 4) 
-- Just 8d

maybeMap :: (a -> b) -> Maybe a -> Maybe b
maybeMap f m = case m of
    Nothing -> Nothing
    Just a -> Just (f a)