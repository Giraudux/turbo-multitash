# turbo-multitash
projet de parallélisation d'algorithme de recherche branch and bound (récursif) de recherche de minimum d'une fonction



Si votre machine a plus de 4 coeurs et 16GB de RAM nous vous conseillons l'utilisation de l'omp imbriqué pour les fonction ayant un seul minimum (booth et beale)

Pour les autres, préférez l'omp simple

Enfin, si vous pouvez travailler sur plusieur machine utilisez MPI
