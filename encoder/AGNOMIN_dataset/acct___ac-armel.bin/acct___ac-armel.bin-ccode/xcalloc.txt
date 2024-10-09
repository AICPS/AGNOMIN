
void * xcalloc(size_t n,size_t s)

{
  void *pvVar1;
  
  pvVar1 = calloc(n,s);
  if (pvVar1 != (void *)0x0) {
    return pvVar1;
  }
                    /* WARNING: Subroutine does not return */
  xalloc_die();
}

