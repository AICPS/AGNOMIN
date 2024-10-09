
void _start(void)

{
  __libc_start_main(main);
  do {
                    /* WARNING: Do nothing block with infinite loop */
  } while( true );
}

