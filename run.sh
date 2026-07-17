for fe_degree in 1 2 3 4 5 6 7 8; do
  echo "*** fe_degree ${fe_degree} ***"

  for ts in -1 32 64 96 128; do
      ./example --degree ${fe_degree}  --bp 1 --ts ${ts}
  done
done
