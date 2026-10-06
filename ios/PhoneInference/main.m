// UIKit host for ggml rpc-server. SpringBoard SIGKILLs a bare main() that never
// finishes launch (~20 s), and iOS 27 UIKit traps without scene lifecycle, so
// UIApplicationMain + a scene delegate own the main thread and the RPC server
// runs on a worker thread. Launch args (DVT) pass through to rpc-server.
#import <UIKit/UIKit.h>
#include <pthread.h>

// rpc-server.cpp is compiled with -Dmain=rpc_server_main, so the symbol is C++-mangled.
int rpc_server_main(int argc, char ** argv) __asm__("__Z15rpc_server_mainiPPc");

static int g_argc;
static char ** g_argv;

static void * serve(void * unused) {
    (void) unused;
    static char * defaults[] = {"rpc-server", "-H", "127.0.0.1", "-p", "50052", "-t", "6", NULL};
    if (g_argc > 1) {
        rpc_server_main(g_argc, g_argv);
    } else {
        rpc_server_main(7, defaults);
    }
    return NULL;
}

@interface SceneDelegate : UIResponder <UIWindowSceneDelegate>
@property (strong, nonatomic) UIWindow * window;
@end

@implementation SceneDelegate
- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)opts {
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *) scene];
    UIViewController * vc = [UIViewController new];
    vc.view.backgroundColor = UIColor.blackColor;
    UILabel * label = [[UILabel alloc] initWithFrame:vc.view.bounds];
    label.text = @"omarchy-cluster\nggml rpc-server :50052";
    label.numberOfLines = 0;
    label.textAlignment = NSTextAlignmentCenter;
    label.textColor = UIColor.whiteColor;
    label.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [vc.view addSubview:label];
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
    app.idleTimerDisabled = YES;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u << 20);
    pthread_t thread;
    pthread_create(&thread, &attr, serve, NULL);
    pthread_attr_destroy(&attr);
    return YES;
}
@end

int main(int argc, char ** argv) {
    g_argc = argc;
    g_argv = argv;
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    }
}
