import Landing from "./Landing";
import Motivations from "./Motivations";
import Features from "./Features";
import DemoLink from "./DemoLink";
import Footer from "./Footer";
import NavBar from "./Navbar";
const Home = () => {
  return (
    <main className="relative min-h-screen w-full">
      <Landing />
      <Motivations />
      <Features />
      <DemoLink />
      <Footer />
    </main>
  );
};

export default Home;
