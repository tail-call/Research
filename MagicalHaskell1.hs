--- # MAGICAL HASKELL Chapter I

--- In ghci, type:

-- :l "MagicalHaskell1.hs"

--- Recursive function example:
fact :: Int -> Int
fact n = if n == 0 then 1 else n * fact (n-1)

--- Loading this file

-- $ ghci
-- GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
-- ghci> :l a.hs
-- [1 of 2] Compiling Main             ( a.hs, interpreted )
-- Ok, one module loaded.

-- ghci> fact 23
-- 8128291617894825984

-- ghci> fact 233
-- 0

-- ghci> fact -2
-- <interactive>:4:6: error:
--     • No instance for (Num (Int -> Int)) arising from a use of ‘-’
--         (maybe you haven't applied a function to enough arguments?)
--     • In the expression: fact - 2
--       In an equation for ‘it’: it = fact - 2


--- Pattern matching declaration example:

fact2 :: Int -> Int
fact2 0 = 1
fact2 n = n * fact2 (n - 1)

-- // patterns are translated into nested case statements
-- // under the hood

--- Fibonacci numbers:

fib :: Int -> Int
fib 0 = 1
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

-- -- function that takes a Char and an Int and returns an int,
-- -- is really a function that takes a Char and returns a function
-- -- that takes an Int and returns an Int
-- anotherWeirdFunction :: Char -> Int -> Int
-- anotherWeirdFunction c i = whichNumberIsChar c + i

-- useless 3 argument multiplication function:
mult3 :: Double -> Double -> Double -> Double
mult3 x y z = x * y * z


whichNumberIsChar :: Char -> Int
whichNumberIsChar = fromEnum :: Char -> Int

-- ghci> whichNumberIsChar 'Я'
-- 1071
-- ghci> whichNumberIsChar '😀'

-- <interactive>:11:15: error:
--     lexical error in string/character literal at end of input


-- List :: * -> *
-- List :: Type -> Type
-- • Invalid type signature: List :: ...


--- ## MULTIPLE ARGUMENTS

--- WRONG WAY: USE A TUPLE

anotherWeirdFunction :: (Char, Int) -> Int
anotherWeirdFunction (c, i) = whichNumberIsChar c + i

-- ghci> anotherWeirdFunction 'a' 1

-- <interactive>:26:1: error:
--     • Couldn't match expected type ‘t0 -> t’ with actual type ‘Int’
--     • The function ‘anotherWeirdFunction’
--       is applied to two value arguments,
--         but its type ‘(Char, Int) -> Int’ has only one
--       In the expression: anotherWeirdFunction 'a' 1
--       In an equation for ‘it’: it = anotherWeirdFunction 'a' 1
--     • Relevant bindings include it :: t (bound at <interactive>:26:1)

-- <interactive>:26:22: error:
--     • Couldn't match expected type ‘(Char, Int)’
--                   with actual type ‘Char’
--     • In the first argument of ‘anotherWeirdFunction’, namely ‘'a'’
--       In the expression: anotherWeirdFunction 'a' 1
--       In an equation for ‘it’: it = anotherWeirdFunction 'a' 1
-- ghci> anotherWeirdFunction ('a', 1)
-- 98

--- Example calls of `map()`:

-- ghci> map (*2) [1,2,3,4]
-- [2,4,6,8]

-- ghci> map (+2) [0,2,4,8]
-- [2,4,6,10]





square :: Num a => a -> a
square x = x * x

-- ghci> square 2
-- 4
-- ghci> square 9
-- 81

polynom2 :: Num a => a -> a -> a
polynom2 x y = square x + 2 * y * x + y

-- ghci> polynom2 2 9
-- 49

-- ghci> polynom2 3 1
-- 16


--- this is mine i wrote this

mul :: (Eq t1, Num t1, Num t2) => t1 -> t2 -> t2
mul 0 y = 0
mul 1 y = y
mul x y = y + mul (x - 1) y


--- Péter and Robinson Ackerman function\
--- <https://www.wikiwand.com/en/articles/Ackermann_function>

ack :: (Num t1, Num t2, Eq t1, Eq t2) => t1 -> t2 -> t2
ack 0 n = n + 1
ack m 0 = ack (m - 1) 1
ack m n = ack (m - 1) (ack m (n - 1))

-- ghci> ack 1 3
-- 5

-- ghci> ack 4 0
-- 13

-- ghci> ack 4 1
-- ^CInterrupted.

