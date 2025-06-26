main :: IO ()
main = ask
    >> getLine
    >>= toList
    >>= calcAverage
    >>= printResult
    where calcAverage l = pure (fromIntegral (sum l) / lengthF l)
          printResult n = putStrLn $ "Average of your list is: " ++ show n
          ask = putStrLn "Enter list of numbers, e.g. [1,3,4,5]:"
          lengthF = fromIntegral . length
          toList :: String -> IO [Int] = pure . read


-- (3.12.3) 0      JupyterNotebooksλ ghc b.hs 
-- [1 of 2] Compiling Main             ( b.hs, b.o )
-- [2 of 2] Linking b
-- ld: warning: ignoring duplicate libraries: '-lm'


-- (3.12.3) 0      JupyterNotebooksλ ./b
-- Enter list of numbers, e.g. [1,3,4,5]:
-- [1,3,4,5]
-- Average of your list is: 3.25