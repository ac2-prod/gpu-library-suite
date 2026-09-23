/* Test-only API subset; no vendor code or production implementation. */
#ifdef __cplusplus
extern "C" {
#endif
void mkl_set_num_threads(int threads);
void mkl_get_version_string(char *text, int length);
#ifdef __cplusplus
}
#endif
