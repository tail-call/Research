--- MAGICAL HASKELL Chapter 2

data JSValue = JString String | JDate Int | JNumber Double | JUndefined
    deriving (Eq, Ord, Show)

valA = JString "Hello"
valB = JDate 404
valC = JNumber 3.33
valD = JUndefined

-- main = print valA >> print valB >> print valC >> print valD

-- 0       JupyterNotebooksλ ghci j.hs
-- GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
-- [1 of 2] Compiling Main             ( j.hs, interpreted )
-- Ok, one module loaded.
-- ghci> main
-- JString "Hello"
-- JDate 404
-- JNumber 3.33
-- JUndefined

data Name = String
data Pony = Unicorn Name | Nonea

--- I call this number for a
data Date = Date {
    year :: Int,
    month :: Int,
    day :: Int
} deriving (Show)

--- These are all countries that exist in the world, all other facts
--- in the Internet about countries are fake
data Country = Equestria | Draconica | Switzerland | MoodDisorderia
    deriving Show

data Address = Address {
    country :: Country,
    city :: String,
    street :: String,
    house :: String
} deriving (Show)

data Person = Person {
    firstName :: String,
    lastName :: String,
    age :: Int,
    dob :: Date,
    address :: Address,
    father :: Maybe Person,
    mother :: Maybe Person
} deriving (Show)

main :: IO ()
main = do
    let person1 = Person {
        firstName = "Twilight",
        lastName = "Sparkle",
        age = 25,
        dob = Date {year = 1990, month = 5, day = 10},
        address = Address {
            country = Equestria,
            city = "Ponyville",
            street = "Main",
            house = "1"
        },
        father = Nothing,
        mother = Nothing
    }

    let person2 = Person {
        firstName = "Princess",
        lastName = "Ember",
        age = 200,
        dob = Date {
            year = 2025 - 200,
            month = 1,
            day = 1
        },
        address = Address {
            country = Draconica,
            city = "Nowhere",
            street = "Master",
            house = "2"
        },
        father = Just person1,
        mother = Just person1
    }

    print person1
    print person2

-- GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
-- [1 of 2] Compiling Main             ( j.hs, interpreted )
-- Ok, one module loaded.
-- ghci> main
-- Person {firstName = "Twilight", lastName = "Sparkle", age = 25, dob = Date {year = 1990, month = 5, day = 10}, address = Address {country = Equestria, city = "Ponyville", street = "Main", house = "1"}, father = Nothing, mother = Nothing}
-- Person {firstName = "Princess", lastName = "Ember", age = 200, dob = Date {year = 1825, month = 1, day = 1}, address = Address {country = Draconica, city = "Nowhere", street = "Master", house = "2"}, father = Just (Person {firstName = "Twilight", lastName = "Sparkle", age = 25, dob = Date {year = 1990, month = 5, day = 10}, address = Address {country = Equestria, city = "Ponyville", street = "Main", house = "1"}, father = Nothing, mother = Nothing}), mother = Just (Person {firstName = "Twilight", lastName = "Sparkle", age = 25, dob = Date {year = 1990, month = 5, day = 10}, address = Address {country = Equestria, city = "Ponyville", street = "Main", house = "1"}, father = Nothing, mother = Nothing})}
-- ghci> 