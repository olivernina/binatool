//#ifndef L_MIN
//#define L_MIN(x,y)   (((x) < (y)) ? (x) : (y))
//#endif
//
//#ifndef L_MAX
//#define L_MAX(x,y)   (((x) > (y)) ? (x) : (y))
//#endif
//
//
///*-------------------------------------------------------------------------*
// *                              Basic Pix                                  *
// *-------------------------------------------------------------------------*/
//struct Pix
//{
//    l_uint32             w;           /* width in pixels                   */
//    l_uint32             h;           /* height in pixels                  */
//    l_uint32             d;           /* depth in bits                     */
//    l_uint32             wpl;         /* 32-bit words/line                 */
//    l_uint32             refcount;    /* reference count (1 if no clones)  */
//    l_uint32             xres;        /* image res (ppi) in x direction    */
//                                      /* (use 0 if unknown)                */
//    l_uint32             yres;        /* image res (ppi) in y direction    */
//                                      /* (use 0 if unknown)                */
//    l_int32              informat;    /* input file format, IFF_*          */
//    char                *text;        /* text string associated with pix   */
//    struct PixColormap  *colormap;    /* colormap (may be null)            */
//    l_uint32            *data;        /* the image data                    */
//};
//typedef struct Pix PIX;
//
//#define CALLOC(numelem, elemsize)   calloc(numelem, elemsize)
//#define FREE(ptr)                   free(ptr)
//
///*-------------------------------------------------------------------------*
// *         Direction flags for grayscale morphology, granulometry,         *
// *                   composable Sels, and convolution                      *
// *-------------------------------------------------------------------------*/
//enum {
//    L_HORIZ            = 1,
//    L_VERT             = 2,
//    L_BOTH_DIRECTIONS  = 3
//};