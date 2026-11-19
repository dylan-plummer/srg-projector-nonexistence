# Strongly regular graphs with parameters (1275,546,281,198) and (1081,594,359,286) do not exist

A short note proving that there are no strongly regular graphs with parameters (1275,546,281,198) or
(1081,594,359,286), and therefore none with the complementary parameters (1275,728,379,464) or (1081,486,177,252).

In both cases the number of vertices equals d(d+1)/2, where d is the multiplicity of the positive eigenvalue. The
vertex projectors of the spherical embedding then form a basis of the symmetric d x d matrices. Expanding a
symmetrised product of two vertex vectors in this basis forces a non-integral number of edges among the common
neighbours of a vertex pair.

| file | contents |
|---|---|
| `note.tex`, `note.pdf` | the note (about 2.5 pages, self-contained for a reader with linear algebra) |
| `verify.py` | exact check of every number in the note, and of the identity on all 5988 vertex pairs of T(5), ..., T(12); Python 3 standard library only |

```
python3 verify.py          # about 10 seconds, ends with "ALL CHECKS PASSED"
pdflatex note && pdflatex note
```
