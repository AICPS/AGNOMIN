
char * xstrdup(char *string)

{
  size_t sVar1;
  char *pcVar2;
  
  sVar1 = strlen(string);
  pcVar2 = (char *)xmemdup(string,sVar1 + 1);
  return pcVar2;
}

