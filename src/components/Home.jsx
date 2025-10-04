import Landing from "./Landing";
import Motivations from "./Motivations";
import Features from "./Features";
import Background from "./background";
import Footer from "./Footer";
import DemoLink from "./DemoLink";

const Home = () => {
  return (
    <main className="relative min-h-screen w-full">
      <Landing />
      <Motivations />
      <Background />
      <Features />
      <DemoLink />
      <Footer />
    </main>
  );
};

export default Home;
